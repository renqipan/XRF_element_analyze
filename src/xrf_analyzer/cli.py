import argparse
import json
import logging
from pathlib import Path
from .config import AnalysisConfig
from .manual_peaks import parse_peak_energies
from .pipeline import analyze
from .reporting import write_json, ANALYSIS_OUTPUT_FILES
from .console import print_failed_summary


def build_parser():
    parser = argparse.ArgumentParser(description='XRF sample spectrum analysis. Sensitivity already encodes the incident excitation spectrum.')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('analyze', help='Analyze raw sample CSV / Amptek ASCII MCA; supports batches')
    p.add_argument('--spectrum', nargs='+', required=True, help='Sample spectra, NOT the incident excitation spectrum')
    p.add_argument('--calibration', required=True)
    p.add_argument('--sensitivity', required=True, help='Nine-branch common-scale S CSV, never row-normalized')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--config', help='JSON overrides of AnalysisConfig; CLI overrides JSON')
    p.add_argument('--background_spectrum','--backgroud_spectrum','--background-spectrum',
                   '--backgroud-spectrum','--background',dest='background_spectrum',default=argparse.SUPPRESS,
                   help='Measured background MCA/CSV; default None; use None to disable; global scale is fitted')
    p.add_argument('--find_peaks', '--find-peaks', nargs='+', default=argparse.SUPPRESS, help='User peak centers in keV; default None (automatic); quote bracket/comma lists')
    p.add_argument('--find-peaks-axis', choices=['reference','observed'], default=argparse.SUPPRESS)
    p.add_argument('--element-prior', choices=['general','heritage','archaeology','minerals','none'], default=argparse.SUPPRESS)
    p.add_argument('--prior-tie-margin', type=float, default=argparse.SUPPRESS)
    p.add_argument('--energy-range', nargs=2, type=float, default=argparse.SUPPRESS)
    p.add_argument('--elements', default=argparse.SUPPRESS, help='Comma-separated canonical symbols')
    p.add_argument('--exclude-elements', default=argparse.SUPPRESS)
    p.add_argument('--reference-element', default=argparse.SUPPRESS)
    p.add_argument('--quantification-method', choices=['local-peaks','global-template'],
                   default=argparse.SUPPRESS, help='Primary peak area / sensitivity (default) or strict global fixed-branch templates')
    p.add_argument('--tube-target', default=argparse.SUPPRESS)
    p.add_argument('--tube-voltage-kv', type=float, default=argparse.SUPPRESS)
    p.add_argument('--scatter-angle-deg', type=float, default=argparse.SUPPRESS)
    p.add_argument('--channel-start', type=int, choices=[0, 1], default=argparse.SUPPRESS)
    p.add_argument('--fwhm-ref-energy-kev', type=float, default=argparse.SUPPRESS)
    p.add_argument('--fwhm-ref-kev', type=float, default=argparse.SUPPRESS)
    p.add_argument('--delta-max-kev', type=float, default=argparse.SUPPRESS)
    p.add_argument('--fano', type=float, default=argparse.SUPPRESS)
    p.add_argument('--pair-energy-kev', type=float, default=argparse.SUPPRESS)
    p.add_argument('--snip-iterations', type=int, default=argparse.SUPPRESS)
    p.add_argument('--background-degree', type=int, default=argparse.SUPPRESS)
    p.add_argument('--candidate-threshold-sigma', type=float, default=argparse.SUPPRESS)
    p.add_argument('--support-delta-deviance', type=float, default=argparse.SUPPRESS)
    p.add_argument('--max-deviance-per-dof', type=float, default=argparse.SUPPRESS)
    p.add_argument('--tentative-delta-deviance', type=float, default=argparse.SUPPRESS)
    p.add_argument('--element-selection', choices=['model-comparison', 'manual'], default=argparse.SUPPRESS)
    p.add_argument('--background-method', choices=['snip'], default='snip')
    p.add_argument('--detector-efficiency', type=float, default=argparse.SUPPRESS)
    p.add_argument('--bootstrap', type=int, default=argparse.SUPPRESS)
    p.add_argument('--seed', type=int, default=argparse.SUPPRESS)
    p.add_argument('--strict-calibration', action='store_true', default=argparse.SUPPRESS)
    p.add_argument('--fill-missing-channels', action='store_true', default=argparse.SUPPRESS)
    p.add_argument('--save-intermediate', action='store_true', default=argparse.SUPPRESS)
    p.add_argument('--no-low-z-kalpha',dest='low_z_kalpha',action='store_false',default=argparse.SUPPRESS,help='Use the stricter two-region identification rule also for Z<=24')
    p.add_argument('--no-atomic-completion', dest='complete_atomic_lines', action='store_false', default=argparse.SUPPRESS)
    p.add_argument('--no-infer-cutoff', dest='infer_low_energy_cutoff', action='store_false', default=argparse.SUPPRESS)
    p.add_argument('--no-estimate-resolution', dest='estimate_resolution_from_peaks', action='store_false', default=argparse.SUPPRESS)
    p.add_argument('--no-fit-gain', dest='fit_energy_gain', action='store_false', default=argparse.SUPPRESS)
    p.add_argument('--alignment-max-shift-kev', type=float, default=argparse.SUPPRESS)
    p.add_argument('--centroid-tolerance-kev', type=float, default=argparse.SUPPRESS)
    p.add_argument('--fixed-response', dest='fit_response', action='store_false', default=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    spectra, output = args.pop('spectrum'), Path(args.pop('output_dir'))
    calibration, sensitivity = args.pop('calibration'), args.pop('sensitivity')
    config_path = args.pop('config')
    args.pop('command')
    args.pop('background_method')
    try:
        settings = json.loads(Path(config_path).read_text(encoding='utf-8-sig')) if config_path else {}
        if 'find_peaks' in args:
            args['find_peaks']=parse_peak_energies(args['find_peaks'])
        settings.update(args)
        for key in ('elements', 'exclude_elements'):
            if isinstance(settings.get(key), str):
                settings[key] = tuple(s.strip() for s in settings[key].split(',') if s.strip())
        config = AnalysisConfig(**settings).validate()
        if len(spectra) > 1 and len({Path(p).stem for p in spectra}) != len(spectra):
            raise ValueError('Batch spectra need unique stems to avoid overwriting outputs')
    except (ValueError, TypeError, OSError) as exc:
        logging.error('%s', exc)
        return 2
    code = 0
    for spectrum in spectra:
        destination = output if len(spectra) == 1 else output/Path(spectrum).stem
        try:
            result = analyze(spectrum, calibration, sensitivity, destination, config)
            if result['summary']['fit_status'] != 'success':
                code = 1
        except (ValueError, OSError, RuntimeError) as exc:
            destination.mkdir(parents=True, exist_ok=True)
            # A failed rerun must not leave successful tables from an earlier run beside its failed summary.
            for name in ANALYSIS_OUTPUT_FILES:
                stale = destination/name
                if stale.is_file():
                    stale.unlink()
            write_json(destination/'fit_summary.json', {'fit_status': 'failed', 'error': str(exc),
                       'input_spectrum': str(Path(spectrum).resolve()), 'parameters': config.to_dict()})
            (destination/(Path(spectrum).stem+'_elements.png')).unlink(missing_ok=True)
            logging.error('%s: %s', spectrum, exc)
            print_failed_summary(spectrum, exc)
            code = 1
    return code
