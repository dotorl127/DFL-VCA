from pathlib import Path
from urllib.parse import quote
import html
import math

from model_utils import get_video_info


def seconds_to_frames(sec: float, fps: float) -> int:
    return max(0, int(round(float(sec) * float(fps))))


def fps_to_rate(fps: float):
    # FCP7 XML represents 29.97 as timebase 30 + ntsc TRUE.
    if abs(fps - 29.97) < 0.08 or abs(fps - 30000/1001) < 0.08:
        return 30, 'TRUE'
    if abs(fps - 59.94) < 0.12 or abs(fps - 60000/1001) < 0.12:
        return 60, 'TRUE'
    return int(round(fps if fps > 0 else 30)), 'FALSE'


def path_to_file_url(path: str) -> str:
    p = Path(path).resolve()
    try:
        return p.as_uri()
    except Exception:
        # Windows-like fallback when generated on Windows.
        s = str(p).replace('\\', '/')
        if len(s) >= 2 and s[1] == ':':
            return 'file://localhost/' + quote(s, safe='/:')
        return 'file://localhost' + quote(s, safe='/:')


def xml_escape(s) -> str:
    return html.escape(str(s), quote=False)


def collect_cut_frames(markers, fps: float, duration_frames: int):
    cuts = {0, duration_frames}
    for m in markers:
        s = seconds_to_frames(m['start'], fps)
        e = seconds_to_frames(m['end'], fps)
        if 0 < s < duration_frames:
            cuts.add(s)
        if 0 < e < duration_frames:
            cuts.add(e)
    return sorted(cuts)


def find_marker_for_segment(markers, seg_start_f: int, seg_end_f: int, fps: float):
    # Attach a label to split clips that overlap marker ranges.
    best = None
    best_ov = 0
    for m in markers:
        ms = seconds_to_frames(m['start'], fps)
        me = seconds_to_frames(m['end'], fps)
        ov = max(0, min(seg_end_f, me) - max(seg_start_f, ms))
        if ov > best_ov:
            best_ov = ov
            best = m
    return best if best_ov > 0 else None


def write_fcp7_xml(out_path: str, video_path: str, markers, sequence_name: str = None):
    info = get_video_info(video_path)
    fps = float(info['fps'] or 30.0)
    timebase, ntsc = fps_to_rate(fps)
    duration_frames = int(info['frame_count'] or seconds_to_frames(info['duration'], fps))
    if duration_frames <= 0:
        duration_frames = max(1, seconds_to_frames(info['duration'], fps))
    width = int(info['width'])
    height = int(info['height'])
    seq_name = sequence_name or (Path(video_path).stem + '_AI_MARKERS')
    file_url = path_to_file_url(video_path)
    file_name = Path(video_path).name
    file_id = 'file-1'

    cuts = collect_cut_frames(markers, fps, duration_frames)
    segments = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]

    def rate_xml(indent=''):
        return f'{indent}<rate>\n{indent}  <timebase>{timebase}</timebase>\n{indent}  <ntsc>{ntsc}</ntsc>\n{indent}</rate>'

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<!DOCTYPE xmeml>')
    lines.append('<xmeml version="4">')
    lines.append('  <sequence id="sequence-1">')
    lines.append(f'    <name>{xml_escape(seq_name)}</name>')
    lines.append(f'    <duration>{duration_frames}</duration>')
    lines.append(rate_xml('    '))
    lines.append('    <timecode>')
    lines.append(rate_xml('      '))
    lines.append('      <string>00:00:00:00</string>')
    lines.append('      <frame>0</frame>')
    lines.append('      <displayformat>NDF</displayformat>')
    lines.append('    </timecode>')
    lines.append('    <media>')
    lines.append('      <video>')
    lines.append('        <format>')
    lines.append('          <samplecharacteristics>')
    lines.append(rate_xml('            '))
    lines.append(f'            <width>{width}</width>')
    lines.append(f'            <height>{height}</height>')
    lines.append('            <anamorphic>FALSE</anamorphic>')
    lines.append('            <pixelaspectratio>square</pixelaspectratio>')
    lines.append('            <fielddominance>none</fielddominance>')
    lines.append('          </samplecharacteristics>')
    lines.append('        </format>')
    lines.append('        <track>')

    for idx, (s, e) in enumerate(segments, 1):
        m = find_marker_for_segment(markers, s, e, fps)
        label = '' if not m else f'_{m["label"]}_{m["score"]:.3f}'
        clip_name = f'{Path(video_path).stem}_{idx:04d}{label}'
        lines.append(f'          <clipitem id="clipitem-{idx}">')
        lines.append(f'            <name>{xml_escape(clip_name)}</name>')
        lines.append(f'            <duration>{duration_frames}</duration>')
        lines.append(rate_xml('            '))
        lines.append(f'            <start>{s}</start>')
        lines.append(f'            <end>{e}</end>')
        lines.append(f'            <in>{s}</in>')
        lines.append(f'            <out>{e}</out>')
        if idx == 1:
            lines.append(f'            <file id="{file_id}">')
            lines.append(f'              <name>{xml_escape(file_name)}</name>')
            lines.append(f'              <pathurl>{xml_escape(file_url)}</pathurl>')
            lines.append(rate_xml('              '))
            lines.append(f'              <duration>{duration_frames}</duration>')
            lines.append('              <media>')
            lines.append('                <video>')
            lines.append('                  <samplecharacteristics>')
            lines.append(rate_xml('                    '))
            lines.append(f'                    <width>{width}</width>')
            lines.append(f'                    <height>{height}</height>')
            lines.append('                    <anamorphic>FALSE</anamorphic>')
            lines.append('                    <pixelaspectratio>square</pixelaspectratio>')
            lines.append('                    <fielddominance>none</fielddominance>')
            lines.append('                  </samplecharacteristics>')
            lines.append('                </video>')
            lines.append('              </media>')
            lines.append('            </file>')
        else:
            lines.append(f'            <file id="{file_id}"/>')
        lines.append('          </clipitem>')
    lines.append('        </track>')
    lines.append('      </video>')
    lines.append('    </media>')

    # Sequence-level ranged markers. Premiere generally retains sequence markers from FCP XML.
    for i, m in enumerate(markers, 1):
        start_f = seconds_to_frames(m['start'], fps)
        end_f = max(start_f + 1, seconds_to_frames(m['end'], fps))
        lines.append('    <marker>')
        lines.append(f'      <name>{xml_escape(m["label"])} #{i:03d} score={m["score"]:.3f}</name>')
        lines.append(f'      <comment>{xml_escape(m.get("start_tc", ""))} - {xml_escape(m.get("end_tc", ""))}</comment>')
        lines.append(f'      <in>{start_f}</in>')
        lines.append(f'      <out>{end_f}</out>')
        lines.append('    </marker>')

    lines.append('  </sequence>')
    lines.append('</xmeml>')

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines), encoding='utf-8')
    return {
        'xml_path': str(out),
        'video_path': str(Path(video_path).resolve()),
        'markers': len(markers),
        'segments': len(segments),
        'fps': fps,
        'duration_frames': duration_frames,
    }
