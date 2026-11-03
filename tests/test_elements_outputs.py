from pathlib import Path
import matplotlib.pyplot as plt
from xrf_analyzer.batch import format_density_summary
from xrf_analyzer.pipeline import analyze
from xrf_analyzer.visualization import make_elements_plot


def test_compact_summary_excludes_pending_density_and_has_four_lines_with_blank_separator():
    samples=[{'Spectrum_File':'sample.mca','Fit_Status':'success','Possible_Elements':'Cr,Pm'}]
    records=[{'Spectrum_File':'sample.mca','Element':'Fe','Element_Status':'supported',
              'Relative_to_Reference':1.,'Reference_Element':'Fe'},
             {'Spectrum_File':'sample.mca','Element':'Cr','Element_Status':'tentative',
              'Relative_to_Reference':999.,'Reference_Element':'Fe'}]
    lines=format_density_summary(samples,records).splitlines()
    assert len(lines)==4
    assert lines[3]==''
    assert lines[0]=='文件名：sample.mca'
    assert 'Fe=1' in lines[1] and 'Cr' not in lines[1] and 'Pm' not in lines[1]
    assert lines[2]=='可能存在的元素：Cr, Pm'


def test_elements_image_contains_only_raw_count_curve_and_supported_labels(synthetic,tmp_path,monkeypatch):
    s,c,t,cfg,_,_=synthetic({'Fe':600.,'Cu':500.})
    result=analyze(s,c,t,config=cfg)
    close=plt.close
    monkeypatch.setattr(plt,'close',lambda *args:None)
    make_elements_plot(result,tmp_path)
    fig=plt.gcf();ax=fig.axes[0]
    assert len(fig.axes)==1 and len(ax.lines)==1
    assert ax.lines[0].get_label()=='Observed raw count'
    labels={name for text in ax.texts for name in text.get_text().split('/')}
    assert {'Fe','Cu'}<=labels
    supported=set(result['identification'].candidates.query("Status=='supported'").Element)
    assert labels<=supported
    assert (tmp_path/(Path(s).stem+'_elements.png')).is_file()
    close(fig)
