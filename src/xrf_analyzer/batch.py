"""Sequential directory analysis with per-sample output and one density TXT.

The long-format density table contains supported-element rows only. Explicit
failure/no-supported rows retain samples that have no defensible numeric ratios.
No physical acquisition condition is inferred from a sample filename.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import shutil
import time
import pandas as pd
from .cli import build_parser as build_analysis_parser
from .config import AnalysisConfig
from .console import print_failed_summary
from .manual_peaks import parse_peak_energies
from .pipeline import analyze
from .quantification import DENSITY_COLUMNS
from .reporting import ANALYSIS_OUTPUT_FILES, sha256, write_json

PREFIX_COLUMNS = ['Spectrum_File', 'Sample_ID', 'Spectrum_SHA256', 'Record_Type',
                  'Fit_Status', 'Analysis_Status']
TAIL_COLUMNS = ['Ratio_Status', 'Ratio_Unavailable_Reason', 'Possible_Elements',
                'Failure_Reason', 'Warnings', 'Sensitivity_Provenance',
                'Excitation_Match_Verified', 'Spectrum_Fit_Image', 'Sample_Output_Directory']
ALL_DENSITY_COLUMNS = PREFIX_COLUMNS + DENSITY_COLUMNS + TAIL_COLUMNS
SUMMARY_COLUMNS = ['Spectrum_File', 'Sample_ID', 'Fit_Status', 'Analysis_Status',
                   'Supported_Elements', 'Possible_Elements', 'Quantitative_Interpretation_Valid',
                   'Failure_Reason', 'Warnings', 'Spectrum_Fit_Image', 'Sample_Output_Directory']


def discover_spectra(input_dir):
    directory = Path(input_dir)
    if not directory.is_dir():
        raise ValueError('Input directory does not exist: ' + str(directory))
    files = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in ('.mca', '.csv')),
                   key=lambda p: (p.name.casefold(), p.name))
    if not files:
        raise ValueError('No MCA or CSV sample spectra found in ' + str(directory))
    return files


def _atomic_txt(path, rows, columns):
    """Write a UTF-8 BOM, tab-separated text table without partial progress files."""
    temporary = path.with_name(path.name + '.tmp')
    pd.DataFrame(rows, columns=columns).to_csv(temporary, sep='\t', index=False, encoding='utf-8-sig')
    temporary.replace(path)


def format_density_summary(samples,records):
    """Three content lines and one blank line per input, including failed inputs."""
    lines=[]
    clean=lambda value:' '.join(str(value).splitlines())
    for sample in samples:
        entries=[]
        references=set()
        for row in records:
            if row['Spectrum_File']!=sample['Spectrum_File'] or row.get('Element_Status')!='supported':continue
            ratio=row.get('Relative_to_Reference')
            text='暂不可估计'
            if ratio is not None and pd.notna(ratio):
                text='%.6g'%ratio
                if row.get('Reference_Element'):references.add(str(row['Reference_Element']))
            entries.append('%s=%s'%(row['Element'],text))
        failed=sample['Fit_Status']!='success'
        prefix='确定元素及相对面密度'+('（%s=1）'%next(iter(references)) if len(references)==1 else '')+'：'
        lines.extend(['文件名：'+clean(sample['Spectrum_File']),
                      prefix+('；'.join(entries) or ('无法判定（分析失败）' if failed else '无')),
                      '可能存在的元素：'+(clean(sample.get('Possible_Elements','')).replace(',',', ') or ('无法判定' if failed else '无')), ''])
    return '\n'.join(lines)+'\n' if lines else ''


def _write_density_summary(path,samples,records):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(format_density_summary(samples,records),encoding='utf-8-sig')
    temporary.replace(path)


def _clean_failed_output(destination):
    for name in ANALYSIS_OUTPUT_FILES:
        stale = destination/name
        if stale.is_file():
            stale.unlink()


def run_batch(input_dir, output_dir, calibration, sensitivity, config=None):
    """Analyze every top-level .MCA/.CSV; persist aggregate progress after each file.

    Returns a manifest. A parse/optimization failure is recorded and later samples
    still run. Optimizer success with model mismatch is not quantitative success.
    """
    cfg = (config or AnalysisConfig(energy_range=(1., 40.), tube_target='Ag')).validate()
    source, out = Path(input_dir).resolve(), Path(output_dir).resolve()
    if source == out:
        raise ValueError('Input and output directories must differ')
    files = discover_spectra(source)
    stems=[p.stem.casefold() for p in files]
    image_keys=[(p.stem.casefold(),p.suffix.lower()) for p in files]
    calibration, sensitivity = Path(calibration).resolve(), Path(sensitivity).resolve()
    for label, path in [('calibration', calibration), ('sensitivity', sensitivity)]:
        if not path.is_file():
            raise ValueError('Missing ' + label + ' file: ' + str(path))
    out.mkdir(parents=True, exist_ok=True)
    # Remove only the former aggregate filenames when migrating an existing run.
    for stale_name in ('relative_areal_density_all.csv', 'batch_summary.csv'):
        (out/stale_name).unlink(missing_ok=True)
    started = time.monotonic()
    density_rows, sample_rows = [], []
    failures = 0
    manifest = {
        'schema_version': '3.0', 'input_directory': str(source), 'output_directory': str(out),
        'calibration': {'path': str(calibration), 'sha256': sha256(calibration)},
        'sensitivity': {'path': str(sensitivity), 'sha256': sha256(sensitivity)},
        'parameters': asdict(cfg), 'sample_count': len(files), 'processed_samples': 0,
        'failures': 0, 'completed': False,
        'physical_conditions_policy': 'No tube voltage, target, or sensitivity matching inferred from filenames',
        'aggregate_density_txt': 'relative_areal_density_all.txt', 'sample_summary_txt': 'batch_summary.txt',
        'text_format': 'UTF-8 BOM; three content lines and one blank line per spectrum',
        'density_details_json': 'density_details.json',
        'samples': sample_rows,
    }
    for index, spectrum in enumerate(files, 1):
        # Preserve the full filename, including extension, to avoid same-stem collisions.
        sample_id = spectrum.name
        destination = out/'spectra'/sample_id
        destination.mkdir(parents=True, exist_ok=True)
        image_dir=out if stems.count(spectrum.stem.casefold())==1 else out/('elements_'+spectrum.suffix[1:].lower())
        if image_keys.count((spectrum.stem.casefold(),spectrum.suffix.lower())) > 1:
            # .csv/.CSV can coexist on case-sensitive systems. Give each its own
            # directory, also keeping exported paths distinct after case folding.
            image_dir=image_dir/('sample_%04d' % index)
        image_dir.mkdir(parents=True,exist_ok=True)
        image_path = image_dir/(spectrum.stem+'_elements.png')
        (out/(sample_id+'.spectrum_fit.png')).unlink(missing_ok=True)
        image_path.unlink(missing_ok=True)
        relative_destination = destination.relative_to(out).as_posix()
        prefix = {'Spectrum_File': spectrum.name, 'Sample_ID': sample_id, 'Spectrum_SHA256': None}
        print('\n[%d/%d] %s' % (index, len(files), spectrum.name), flush=True)
        try:
            prefix['Spectrum_SHA256'] = sha256(spectrum)
            result = analyze(spectrum, calibration, sensitivity, destination, cfg)
            report = result['summary']
            candidates = result['identification'].candidates
            supported = candidates.loc[candidates.Status == 'supported', 'Element'].tolist()
            possible = report.get('elements_need_confirmation', [])
            successful_fit = report['fit_status'] == 'success'
            figure = destination/(spectrum.stem+'_elements.png')
            if figure.is_file():
                shutil.copyfile(figure, image_path)
            else:
                raise OSError('Analysis did not generate the elements image')
            warnings = '; '.join(str(w) for w in report.get('warnings', []))
            reason = '' if successful_fit else 'joint_optimization_failed'
            tail = {'Failure_Reason': reason, 'Warnings': warnings,
                    'Possible_Elements': ','.join(possible),
                    'Sensitivity_Provenance': report.get('sensitivity', {}).get('sensitivity_provenance'),
                    'Excitation_Match_Verified': report.get('excitation_match_verified', False),
                    'Spectrum_Fit_Image': image_path.relative_to(out).as_posix(),
                    'Sample_Output_Directory': relative_destination}
            prefix.update(Fit_Status=report['fit_status'], Analysis_Status=report.get('analysis_status'))
            # Defense in depth: no tentative/ambiguous element can enter this table,
            # even if a caller supplies a broader diagnostic frame as result['density'].
            densities = result['density']
            densities = densities.loc[densities.Element_Status == 'supported'] if not densities.empty else densities
            if not successful_fit:
                densities = densities.iloc[:0]
            if not densities.empty:
                for record in densities.to_dict('records'):
                    available = pd.notna(record.get('Relative_to_Reference'))
                    unavailable_reason = record.get('Quantification_Status', 'not_estimable')
                    if unavailable_reason=='local_area_not_estimable' and record.get('Measurement_Failure_Reason'):
                        unavailable_reason=record['Measurement_Failure_Reason']
                    if unavailable_reason == 'supported':
                        unavailable_reason = 'no_reliable_supported_reference'
                    density_rows.append({**prefix, 'Record_Type': 'supported_element', **record, **tail,
                        'Ratio_Status': 'available' if available else 'unavailable',
                        'Ratio_Unavailable_Reason': '' if available else unavailable_reason})
            else:
                density_rows.append({**prefix, 'Record_Type': 'no_supported_elements' if successful_fit else 'failed',
                                     **tail, 'Failure_Reason': reason or 'no_supported_elements',
                                     'Ratio_Status': 'unavailable',
                                     'Ratio_Unavailable_Reason': reason or 'no_supported_elements'})
            sample_rows.append({'Spectrum_File': spectrum.name, 'Sample_ID': sample_id,
                                'Fit_Status': report['fit_status'], 'Analysis_Status': report.get('analysis_status'),
                                'Supported_Elements': ','.join(supported), 'Possible_Elements': ','.join(possible),
                                'Quantitative_Interpretation_Valid': report.get('quantitative_interpretation_valid', False),
                                **{key: tail[key] for key in ['Failure_Reason', 'Warnings', 'Spectrum_Fit_Image', 'Sample_Output_Directory']}})
            if not successful_fit:
                failures += 1
        except Exception as exc:
            # Per-file isolation is intentional; KeyboardInterrupt/SystemExit are not swallowed.
            failures += 1
            logging.error('%s: %s', spectrum.name, exc)
            print_failed_summary(spectrum, exc)
            image_path.unlink(missing_ok=True)
            _clean_failed_output(destination)
            (destination/(spectrum.stem+'_elements.png')).unlink(missing_ok=True)
            write_json(destination/'fit_summary.json', {
                'fit_status': 'failed', 'analysis_status': 'input_or_runtime_failed', 'error': str(exc),
                'input_spectrum': str(spectrum), 'parameters': asdict(cfg)})
            prefix.update(Fit_Status='failed', Analysis_Status='input_or_runtime_failed')
            tail = {'Failure_Reason': str(exc), 'Warnings': '', 'Spectrum_Fit_Image': '',
                    'Sample_Output_Directory': relative_destination}
            density_rows.append({**prefix, 'Record_Type': 'failed', **tail,
                                'Ratio_Status': 'unavailable', 'Ratio_Unavailable_Reason': 'input_or_runtime_failed'})
            sample_rows.append({'Spectrum_File': spectrum.name, 'Sample_ID': sample_id,
                                'Fit_Status': 'failed', 'Analysis_Status': 'input_or_runtime_failed',
                                'Supported_Elements': '', 'Possible_Elements': '',
                                'Quantitative_Interpretation_Valid': False, **tail})
        manifest.update(processed_samples=index, failures=failures, elapsed_seconds=time.monotonic()-started)
        _write_density_summary(out/'relative_areal_density_all.txt',sample_rows,density_rows)
        write_json(out/'density_details.json',{'columns':ALL_DENSITY_COLUMNS,'records':density_rows})
        _atomic_txt(out/'batch_summary.txt', sample_rows, SUMMARY_COLUMNS)
        write_json(out/'batch_manifest.json', manifest)
    manifest.update(completed=True, successful_samples=len(files)-failures)
    write_json(out/'batch_manifest.json', manifest)
    print('\n批量处理完成：%d 个样品，%d 个拟合/输入失败。' % (len(files), failures), flush=True)
    print('统一面密度TXT：'+str(out/'relative_areal_density_all.txt'), flush=True)
    print('样品状态TXT：'+str(out/'batch_summary.txt'), flush=True)
    print('“暂不可估计”表示未通过定量质量检查，不代表零含量。', flush=True)
    return manifest


def _configuration_from_analysis_cli(calibration, sensitivity, output_dir, config_path, forwarded):
    # Reuse the current single-spectrum parser so newly added analysis parameters
    # remain available without maintaining a second diverging option inventory.
    argv = ['analyze', '--spectrum', '__batch_sample__', '--calibration', str(calibration),
            '--sensitivity', str(sensitivity), '--output-dir', str(output_dir)]
    if config_path:
        argv += ['--config', str(config_path)]
    args = vars(build_analysis_parser().parse_args(argv+forwarded))
    settings = {'energy_range': (1., 40.), 'tube_target': 'Ag'}
    if args['config']:
        settings.update(json.loads(Path(args['config']).read_text(encoding='utf-8-sig')))
    for key in ('command', 'spectrum', 'calibration', 'sensitivity', 'output_dir', 'config', 'background_method'):
        args.pop(key, None)
    if 'find_peaks' in args:
        args['find_peaks'] = parse_peak_energies(args['find_peaks'])
    settings.update(args)
    for key in ('elements', 'exclude_elements'):
        if isinstance(settings.get(key), str):
            settings[key] = tuple(s.strip() for s in settings[key].split(',') if s.strip())
    return AnalysisConfig(**settings).validate()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Analyze a directory of raw MCA/CSV sample spectra sequentially.',
                                     epilog='Tube target defaults to Ag; override with --tube-target Cu. Other analysis options are forwarded to xrf_analyzer analyze.')
    parser.add_argument('--input-dir', default='XRF_SDD')
    parser.add_argument('--output-dir', default='XRF_Analyze_results')
    parser.add_argument('--calibration', default='data/calibration.csv')
    parser.add_argument('--sensitivity', default='data/xrf_relative_sensitivity.csv')
    parser.add_argument('--config', default=None)
    args, forwarded = parser.parse_known_args(argv)
    # Keep analysis diagnostics in saved reports without repeating them on screen.
    console = logging.StreamHandler()
    console.addFilter(lambda record: record.levelno != logging.WARNING
                      or not getattr(record, 'analysis_diagnostic', False))
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s', handlers=[console])
    try:
        config = _configuration_from_analysis_cli(args.calibration, args.sensitivity, args.output_dir,
                                                   args.config, forwarded)
        manifest = run_batch(args.input_dir, args.output_dir, args.calibration, args.sensitivity, config)
    except (ValueError, TypeError, OSError) as exc:
        logging.error('%s', exc)
        return 2
    return 1 if manifest['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
