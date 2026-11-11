"""Create a self-contained 3D snapshot of closed cabinet grasp sequences."""
import argparse
import base64
from collections import Counter
from datetime import datetime
from html.parser import HTMLParser
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

TEMPLATES = ('xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx', 'post_grasp', 'mid_lift', 'pre_carry')
OUTCOMES = ('complete', 'grasp_then_failed', 'no_grasp')
COLORS = {'complete': '#16804a', 'grasp_then_failed': '#783cad', 'no_grasp': '#db7813'}
LABELS = {'complete': 'Complete pipeline', 'grasp_then_failed': 'Confirmed grasp; pipeline failed',
          'no_grasp': 'No confirmed grasp'}
DEFAULT_PLOTLY = Path('artifacts/thor_dynamic_coverage_graph/coverage_network.html')


class ScriptReader(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.active = False
        self.parts = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.active = True
            self.parts = []

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == 'script' and self.active:
            self.scripts.append(''.join(self.parts))
            self.active = False


def plotly_bundle(path):
    parser = ScriptReader()
    parser.feed(path.read_text(encoding='utf-8'))
    scripts = [script for script in parser.scripts if 'plotly.js v' in script[:256]]
    if len(scripts) != 1:
        raise ValueError('Expected one inline licensed plotly.js bundle')
    return scripts[0]


def json_script(value):
    # JSON must not be able to terminate its containing script element.
    return json.dumps(value, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def freeze_closed(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('protocol') != 'paired_axis_order_and_grasp_insertion_v1':
        raise ValueError('Expected paired cabinet grasp sequences manifest')
    closed = []
    for case in manifest['cases']:
        summary = directory/'cases'/f"case_{case['case_id']:03d}"/'summary.json'
        if summary.is_file():
            closed.append((case, summary))
    return manifest, closed


def read_case(case, summary_path):
    summary = json.loads(summary_path.read_text())
    if summary['case'] != case:
        raise ValueError(f'Summary case mismatch: {summary_path}')
    attempt = summary['attempt']
    directory = summary_path.parent/f'attempt_{attempt:06d}'
    skills = [json.loads(line) for line in (directory/'skills.jsonl').read_text().splitlines() if line.strip()]
    rows = [json.loads(line) for line in (directory/'transitions.jsonl').read_text().splitlines() if line.strip()]
    if not skills:
        raise ValueError(f'Closed case lacks an initial skill: {summary_path}')
    if len(rows) != summary['control_ticks']:
        raise ValueError(f'Closed case tick count mismatch: {summary_path}')
    trace_path = directory/'phase_trace.json'
    if not trace_path.is_file():
        raise ValueError(f'Missing phase trace: {trace_path}')
    trace = json.loads(trace_path.read_text())
    if len(trace) != len(rows) or any(t['step'] != i or row['step'] != i for i, (t, row) in enumerate(zip(trace, rows), 1)):
        raise ValueError(f'Phase trace step count or ordering mismatch: {trace_path}')
    initial = np.asarray(skills[0]['state'], dtype=float)
    if initial.shape != (67,) or not np.isfinite(initial).all():
        raise ValueError(f'Invalid initial state: {summary_path}')
    states = [initial]
    for row in rows:
        before = np.asarray(row['state'], dtype=float)
        after = np.asarray(row['next_state'], dtype=float)
        if (row['episode_id'] != 1 or before.shape != (67,) or after.shape != (67,) or
                not np.isfinite(after).all() or not np.array_equal(before, states[-1])):
            raise ValueError(f'Invalid or discontinuous transition: {summary_path}')
        states.append(after)
    states = np.asarray(states)
    obj_slice = slice(33, 36) if case['object'] == 'cup' else slice(45, 48)
    tcp, target = states[:, 18:21], states[:, obj_slice]
    length = lambda path: float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    outcome = 'complete' if summary['success'] else ('grasp_then_failed' if summary['report']['ever_confirmed_grasp'] else 'no_grasp')
    report = summary['report']
    metric_names = ('ever_confirmed_grasp', 'max_stable_grasp_ticks', 'held_lift_verified',
                    'max_root_rise_while_confirmed_held_m', 'max_continuous_grasp_rise_m',
                    'carry_displacement_m', 'released_after_drop', 'failure')
    metrics = {name: report[name] for name in metric_names}
    metrics['calibration_success'] = summary['success']
    metrics['primitive_statuses'] = [dict(phase=command['phase'], primitive=command['primitive'],
                                          status=command['status']) for command in report['commands']
                                     if command['status'] != 'requested']
    steps = np.asarray([0] + [t['step'] for t in trace])
    phases = np.asarray(['reset'] + [t['phase'] for t in trace])
    target_grasped = np.asarray([False] + [bool(t['target_grasped']) for t in trace])
    stable_ticks = np.asarray([0] + [t['stable_grasp_ticks'] for t in trace])
    closed_command = np.asarray([False] + [bool(t['closed_command']) for t in trace])
    confirmed_grasp = target_grasped & (stable_ticks >= 4)
    return dict(case_id=case['case_id'], object=case['object'], template=case['template'], outcome=outcome,
                attempt=attempt, summary_path=str(summary_path), transitions_path=str(directory/'transitions.jsonl'),
                skills_path=str(directory/'skills.jsonl'), raw_ticks=len(rows), raw_points=len(states),
                tcp_path_length_m=length(tcp), target_path_length_m=length(target),
                tcp=tcp, target=target, initial_cup=initial[33:36], initial_bowl=initial[45:48],
                metrics=metrics, trace_path=str(trace_path), step=steps, phase=phases,
                target_grasped=target_grasped, stable_grasp_ticks=stable_ticks, closed_command=closed_command,
                confirmed_grasp=confirmed_grasp, target_height_relative_initial_m=target[:, 2]-target[0, 2])


def voxel_keys(points, size):
    if not np.isfinite(size) or size <= 0:
        raise ValueError('Voxel size must be finite and positive')
    return np.unique(np.floor(np.asarray(points, dtype=float)/size).astype(np.int64), axis=0).tolist()


def add_voxels(record, size):
    for kind in ('tcp', 'target'):
        record[kind+'_voxels'] = voxel_keys(record[kind], size)


def voxel_visits(records, kind):
    counts = Counter()
    for record in records:
        counts.update({tuple(key) for key in record[kind+'_voxels']})
    return counts


def density_figure(path, records, size):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    for index, kind in enumerate(('tcp', 'target')):
        ax = fig.add_subplot(1, 2, index+1, projection='3d')
        visits = voxel_visits(records, kind)
        if visits:
            centres = (np.asarray(list(visits), dtype=float)+.5)*size
            counts = np.asarray(list(visits.values()))
            scatter = ax.scatter(*centres.T, c=counts, cmap='viridis',
                                 norm=LogNorm(vmin=1, vmax=max(2, int(counts.max()))), s=12,
                                 depthshade=False)
            bar = fig.colorbar(scatter, ax=ax, shrink=.6, pad=.1)
            bar.set_label('Distinct cases (log scale)')
            lo, hi = centres.min(axis=0), centres.max(axis=0)
            lo[2], hi[2] = min(lo[2], .672), max(hi[2], .672)
            lo -= size; hi += size
            ax.plot([lo[0], hi[0]], [lo[1], lo[1]], [.672, .672], color='#988b7b', ls=':', lw=1)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect(hi-lo)
        else:
            ax.text2D(.3, .5, 'No closed cases', transform=ax.transAxes)
        ax.set_title('TCP' if kind == 'tcp' else 'Target object')
        ax.set_xlabel('Root X (m)'); ax.set_ylabel('Root Y (m)'); ax.set_zlabel('Root Z (m)')
    fig.suptitle(f'Recorded-position visitation · unique cases per {size:g} m cell · includes initial positions')
    fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)


def display_case(record, max_points=128):
    count = len(record['tcp'])
    indices = np.linspace(0, count-1, min(count, max_points), dtype=int)
    return {key: (record[key][indices].tolist() if key in ('tcp', 'target', 'step', 'phase', 'target_grasped', 'stable_grasp_ticks',
                                                             'closed_command', 'confirmed_grasp', 'target_height_relative_initial_m') else
                  record[key].tolist() if key in ('initial_cup', 'initial_bowl') else record[key])
            for key in record}


def overview(path, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(15, 13), constrained_layout=True)
    for index, template in enumerate(TEMPLATES):
        ax = fig.add_subplot(3, 3, index+1, projection='3d')
        examples = [next((r for r in records if r['template'] == template and r['outcome'] == outcome), None)
                    for outcome in OUTCOMES]
        plotted = []
        for r in filter(None, examples):
            color = COLORS[r['outcome']]
            tcp, target = r['tcp'], r['target']
            ax.plot(*tcp.T, color=color, lw=1.5)
            ax.plot(*target.T, color=color, ls=':', lw=1.2)
            ax.scatter(*tcp[0], color=color, marker='o', s=12)
            ax.scatter(*tcp[-1], color=color, marker='s', s=12)
            ax.scatter(*r['initial_cup'], color='#333333', marker='^', s=12)
            ax.scatter(*r['initial_bowl'], color='#333333', marker='v', s=12)
            plotted.extend([tcp, target, r['initial_cup'][None], r['initial_bowl'][None]])
        if plotted:
            points = np.concatenate(plotted)
            low, high = points.min(axis=0), points.max(axis=0)
            low[2], high[2] = min(low[2], .672), max(high[2], .672)
            low -= .02; high += .02
            ax.plot([low[0], high[0]], [low[1], low[1]], [.672, .672], color='#988b7b', ls=':', lw=1)
            ax.set_xlim(low[0], high[0]); ax.set_ylim(low[1], high[1]); ax.set_zlim(low[2], high[2])
            ax.set_box_aspect(high-low)
        ax.set_title(template.replace('_', ' '), fontsize=10)
        ax.set_xlabel('X (m)', fontsize=7); ax.set_ylabel('Y (m)', fontsize=7); ax.set_zlabel('Z (m)', fontsize=7)
        ax.tick_params(labelsize=6)
    handles = [Line2D([0], [0], color=COLORS[o], label=LABELS[o]) for o in OUTCOMES]
    handles += [Line2D([0], [0], color='#333', ls='-', label='TCP'),
                Line2D([0], [0], color='#333', ls=':', label='Target object'),
                Line2D([0], [0], color='#333', marker='o', ls='', label='TCP start'),
                Line2D([0], [0], color='#333', marker='s', ls='', label='TCP end'),
                Line2D([0], [0], color='#333', marker='^', ls='', label='Initial cup / bowl (▲ / ▼)'),
                Line2D([0], [0], color='#988b7b', ls=':', label='Shelf height')]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, fontsize=8)
    fig.suptitle('Representative trajectories: at most one case per outcome and template', fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


HTML = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Cabinet grasp sequence snapshot</title>
<style>body{font:15px system-ui;color:#223047;margin:0;background:#f6f8fb}main{max-width:1500px;margin:auto;padding:18px}h1{margin:0 0 6px}p{margin:6px 0}.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:end;margin:14px 0}label{display:grid;gap:4px}select{padding:6px;font:inherit}#plot{height:75vh;min-height:520px;background:white}.note{color:#516175}#counts{font-weight:650}.cards{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}.card{background:white;border:1px solid #d9e1ed;border-radius:8px;padding:10px;white-space:pre-line;flex:1;min-width:220px}#step{width:min(500px,80vw)}</style></head><body><main>
<h1>Cabinet grasp sequences · 3D snapshot</h1><p class="note">Frozen closed cases at __TIME__ (Europe/London); not live. Robot-root coordinates in metres. TCP paths do not certify whole-arm collision clearance.</p>
<div class="controls"><label>View<select id="view"><option value="density">Density</option><option value="paths">Paths</option></select></label><label>Density positions<select id="density_kind"><option value="tcp">TCP</option><option value="target">Target object</option></select></label><label>Object<select id="object"><option value="all">All</option><option value="cup">Cup</option><option value="bowl">Bowl</option></select></label>
<label>Template<select id="template"><option value="all">All</option></select></label><label>Outcome<select id="outcome"><option value="all">All</option></select></label>
<label>Case<select id="case"><option value="all">All</option></select></label><label><span>Cabinet context</span><input type="checkbox" id="cabinet_context" checked></label><label><span>Paths</span><span><input id="tcp" type="checkbox" checked> TCP <input id="target" type="checkbox" checked> Target object</span></label></div>
<p id="counts"></p><p id="grasp_counts"></p><div class="cards"><div class="card" id="case_metrics"></div><div class="card" id="case_motion"></div><div class="card" id="case_primitives"></div></div><label>Inspect recorded control step (select one case and Paths) <input id="step" type="range" min="0" max="0" value="0" disabled></label><div><button id="path_play" disabled>Play</button> <label style="display:inline">Path speed <select id="path_speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1× (real time)</option><option value="2">2×</option></select></label></div><p id="step_label" class="note"></p><p class="note">Recorded-position visitation; distinct cases per cell; each case counted once; gaps between samples not filled. Includes initial positions, so common starts appear hot. Markers show cell centres, not filled volumes. Cell width: __VOXEL_SIZE__ m.</p><p class="note">Single-case Paths: full 20 Hz recorded TCP path colored by phase, dotted dark teal target-object path and arrows showing movement direction; start and end are labeled. Cabinet wireframe and shelf are schematic, without the door or robot collision geometry. Every recorded controller step is retained. The single-case view starts zoomed around the route; zoom out to see the whole cabinet.</p>
<div id="plot"></div></main><script>__PLOTLY__</script><script>const DATA=__DATA__; const COLORS=__COLORS__; const LABELS=__LABELS__; const TEMPLATES=__TEMPLATES__;
const el=id=>document.getElementById(id); const select=(id,v,label)=>el(id).add(new Option(label,v));
TEMPLATES.forEach(x=>select('template',x,x.replaceAll('_',' '))); Object.entries(LABELS).forEach(([v,l])=>select('outcome',v,l));
DATA.cases.forEach(r=>select('case',String(r.case_id),`${r.case_id} · ${r.object} · ${r.template} · ${LABELS[r.outcome]}`));
function render(){let rows=DATA.cases.filter(r=>['object','template','outcome','case'].every(k=>el(k).value==='all'||String(r[k==='case'?'case_id':k])===el(k).value));
const counts=Object.fromEntries(Object.keys(LABELS).map(o=>[o,rows.filter(r=>r.outcome===o).length]));el('counts').textContent=`${rows.length} matched of ${DATA.cases.length} closed cases · ${Object.entries(LABELS).map(([o,l])=>`${l}: ${counts[o]}`).join(' · ')} · ${DATA.unfinished} unfinished at capture`;
el('grasp_counts').textContent=`Confirmed grasp: ${rows.filter(r=>r.metrics.ever_confirmed_grasp).length} · Verified held lift: ${rows.filter(r=>r.metrics.held_lift_verified).length} · Full pipeline: ${rows.filter(r=>r.metrics.calibration_success).length}`;
let single=el('case').value!=='all'&&rows.length===1?rows[0]:null;let slider=el('step');slider.disabled=!(single&&el('view').value==='paths');el('path_play').disabled=slider.disabled;
if(single){let m=single.metrics;el('case_metrics').textContent=`Case ${single.case_id} · ${single.object} · ${single.template}\nConfirmed grasp: ${m.ever_confirmed_grasp?'yes':'no'} · maximum stable ticks: ${m.max_stable_grasp_ticks}\nVerified held lift: ${m.held_lift_verified?'yes':'no'}\nFull pipeline: ${m.calibration_success?'yes':'no'} · released after drop: ${m.released_after_drop?'yes':'no'}`;
el('case_motion').textContent=`Held rise from initial object position: ${(100*m.max_root_rise_while_confirmed_held_m).toFixed(2)} cm\nContinuous grasp rise: ${(100*m.max_continuous_grasp_rise_m).toFixed(2)} cm\nCarry displacement: ${(100*m.carry_displacement_m).toFixed(2)} cm\nFailure: ${m.failure??'none'}`;
el('case_primitives').textContent=['Primitive sequence / status',...m.primitive_statuses.map(p=>`${p.phase}: ${p.primitive} · ${p.status}`)].join(String.fromCharCode(10));
if(slider.dataset.caseId!==String(single.case_id)){slider.value=0;slider.dataset.caseId=String(single.case_id)}slider.max=single.step.length-1;
let i=Math.min(Number(slider.value),single.step.length-1);let feedback=i===0?'reset: no grasp feedback':`phase ${single.phase[i]} · confirmed grasp ${single.confirmed_grasp[i]?'yes':'no'} · target height change ${(100*single.target_height_relative_initial_m[i]).toFixed(2)} cm`;
el('step_label').textContent=`Recorded control step ${single.step[i]} · ${(single.step[i]/20).toFixed(2)} s · ${feedback}${slider.disabled?' · switch to Paths to inspect':''}`;
}else{slider.value=0;slider.max=0;slider.dataset.caseId='';el('step_label').textContent='Select one matching case and Paths to inspect sampled steps.';el('case_metrics').textContent=rows.length?'Select one case for grasp and pipeline evidence.':'No cases match these filters.';el('case_motion').textContent='Measured held rise and carry displacement appear for one case.';el('case_primitives').textContent='Primitive statuses, including timeouts, appear for one case.';}
if(el('view').value==='density'){let visits=new Map();let kind=el('density_kind').value;for(const r of rows)for(const key of r[kind+'_voxels']){let k=key.join(',');visits.set(k,(visits.get(k)||0)+1)}
let keys=Array.from(visits.keys(),k=>k.split(',').map(Number)), values=Array.from(visits.values());let max=values.reduce((a,b)=>Math.max(a,b),0);let ticks=[];for(let n=1;n<=max;n*=10)ticks.push(n);if(max&&!ticks.includes(max))ticks.push(max);
let trace={type:'scatter3d',mode:'markers',x:keys.map(k=>(k[0]+.5)*DATA.voxel_size),y:keys.map(k=>(k[1]+.5)*DATA.voxel_size),z:keys.map(k=>(k[2]+.5)*DATA.voxel_size),customdata:values.map(v=>[v,rows.length?v/rows.length:0]),name:'Visited cell centres',marker:{size:4,color:values.map(Math.log10),colorscale:'Viridis',cmin:0,cmax:Math.max(1,Math.log10(max||1)),colorbar:{title:'Cases<br>(log scale)',tickvals:ticks.map(Math.log10),ticktext:ticks.map(String)}},hovertemplate:'x %{x:.3f} y %{y:.3f} z %{z:.3f}<br>%{customdata[0]} distinct cases<br>%{customdata[1]:.1%} of selected cases<extra></extra>'};
let traces=[trace];if(keys.length){let xs=keys.map(k=>(k[0]+.5)*DATA.voxel_size),ys=keys.map(k=>(k[1]+.5)*DATA.voxel_size);let xmin=xs.reduce((a,b)=>Math.min(a,b),Infinity),xmax=xs.reduce((a,b)=>Math.max(a,b),-Infinity),y=ys.reduce((a,b)=>Math.min(a,b),Infinity);traces.push({type:'scatter3d',mode:'lines',x:[xmin,xmax],y:[y,y],z:[.672,.672],name:'Shelf height',line:{color:'#988b7b',dash:'dot',width:3},hoverinfo:'name'});}
Plotly.react('plot',traces,{margin:{l:0,r:0,t:10,b:0},scene:{xaxis:{title:'Root X (m)'},yaxis:{title:'Root Y (m)'},zaxis:{title:'Root Z (m)'},aspectmode:'data'},showlegend:false,annotations:rows.length?[]:[{text:'No cases match these filters',showarrow:false}],uirevision:'snapshot'}, {responsive:true,displaylogo:false});return;}
let traces=[];if(!single)for(const [outcome,label] of Object.entries(LABELS)){let group=rows.filter(r=>r.outcome===outcome);for(const kind of ['tcp','target']){if(!el(kind).checked)continue;
let x=[],y=[],z=[],custom=[];for(const r of group){for(let i=0;i<r[kind].length;i++){let p=r[kind][i];x.push(p[0]);y.push(p[1]);z.push(p[2]);custom.push(`${r.case_id} · ${r.object} · ${r.template} · ${LABELS[r.outcome]} · step ${r.step[i]} · ${r.phase[i]} · ${i===0?'reset; no feedback':r.confirmed_grasp[i]?'confirmed grasp':'unconfirmed grasp'}`)}x.push(null);y.push(null);z.push(null);custom.push(null)}
traces.push({type:'scatter3d',mode:'lines',x,y,z,customdata:custom,name:`${label} · ${kind==='tcp'?'TCP':'object'}`,line:{color:COLORS[outcome],width:kind==='tcp'?4:3,dash:kind==='tcp'?'solid':'dot'},hovertemplate:'%{customdata}<br>x %{x:.4f} y %{y:.4f} z %{z:.4f}<extra></extra>'});}
if(el('tcp').checked){for(const [ix,labelMark,symbol] of [[0,'start','circle'],[-1,'end','square']]){let points=group.map(r=>r.tcp.at(ix));traces.push({type:'scatter3d',mode:'markers',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),customdata:group.map(r=>`${r.case_id} · ${r.object} · ${r.template} · ${LABELS[r.outcome]}`),name:`${label} · ${labelMark}`,marker:{color:COLORS[outcome],size:3,symbol},hovertemplate:'%{customdata}<extra></extra>'});}}}
if(single){
let palette=['#1765ad','#e46c0a','#8b3fb0','#188453','#db334d','#007e91','#967000','#ce53a0','#4d66b2','#785445','#719c30','#af502f'];let phases=[...new Set(single.phase)];let phaseColors=Object.fromEntries(phases.map((p,i)=>[p,palette[i%palette.length]]));
if(el('tcp').checked){for(let start=0;start<single.tcp.length;){let phase=single.phase[start],end=start+1;while(end<single.tcp.length&&single.phase[end]===phase)end++;let points=single.tcp.slice(Math.max(0,start-1),end);let meta=single.step.slice(Math.max(0,start-1),end).map((step,i)=>{let n=Math.max(0,start-1)+i;return `step ${step} · ${single.phase[n]} · ${n===0?'reset; no feedback':single.confirmed_grasp[n]?'confirmed grasp':'unconfirmed grasp'}`});traces.push({type:'scatter3d',mode:'lines',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),customdata:meta,name:`TCP · ${phase}`,line:{color:phaseColors[phase],width:6},hovertemplate:'%{customdata}<br>x %{x:.4f} y %{y:.4f} z %{z:.4f}<extra></extra>'});start=end;}
let candidates=[];for(let j=1;j<single.tcp.length;j++){let a=single.tcp[j-1],b=single.tcp[j],v=b.map((n,k)=>n-a[k]);let norm=Math.hypot(...v);if(norm>1e-6)candidates.push({p:b,v:v.map(q=>q/norm)});}let stride=Math.max(1,Math.ceil(candidates.length/12)),indices=candidates.filter((_,i)=>i%stride===0);if(indices.length)traces.push({type:'cone',x:indices.map(a=>a.p[0]),y:indices.map(a=>a.p[1]),z:indices.map(a=>a.p[2]),u:indices.map(a=>a.v[0]),v:indices.map(a=>a.v[1]),w:indices.map(a=>a.v[2]),anchor:'tip',sizemode:'absolute',sizeref:.018,colorscale:[[0,'#222222'],[1,'#222222']],showscale:false,name:'Direction',hoverinfo:'skip'});
let points=[single.tcp[0],single.tcp.at(-1)];traces.push({type:'scatter3d',mode:'markers+text',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),text:['Start','End'],textposition:'top center',marker:{color:['#111111','#111111'],size:5,symbol:['circle','square']},name:'Start / end',hoverinfo:'text'});}
if(el('target').checked){let points=single.target;traces.push({type:'scatter3d',mode:'lines',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),name:'Target object path',line:{color:'#005f60',width:4,dash:'dot'},hoverinfo:'name'});}
if(el('cabinet_context').checked){let x=[.275,.775],y=[-.18,.42],z=[.30,1.02],xs=[],ys=[],zs=[];let edge=(a,b)=>{xs.push(a[0],b[0],null);ys.push(a[1],b[1],null);zs.push(a[2],b[2],null)};for(let yy of y)for(let zz of z)edge([x[0],yy,zz],[x[1],yy,zz]);for(let xx of x)for(let zz of z)edge([xx,y[0],zz],[xx,y[1],zz]);for(let xx of x)for(let yy of y)edge([xx,yy,z[0]],[xx,yy,z[1]]);traces.push({type:'scatter3d',mode:'lines',x:xs,y:ys,z:zs,name:'Cabinet frame (schematic)',line:{color:'#8e9298',width:3},hoverinfo:'skip'});traces.push({type:'mesh3d',x:[.29,.76,.76,.29],y:[-.15,-.15,.39,.39],z:[.672,.672,.672,.672],i:[0,0],j:[1,2],k:[2,3],opacity:.15,color:'#a68159',name:'Shelf (schematic)',hoverinfo:'skip'});}}
if(single){let i=Math.min(Number(slider.value),single.step.length-1);let color=i>0&&single.confirmed_grasp[i]?'#16804a':'#808080';for(const [kind,symbol,label] of [['tcp','circle','Current TCP'],['target','diamond','Current target']]){let p=single[kind][i];traces.push({type:'scatter3d',mode:'markers',x:[p[0]],y:[p[1]],z:[p[2]],name:label,marker:{color,size:7,symbol},hovertemplate:label+'<extra></extra>'});}}
for(const [key,label,symbol] of [['initial_cup','Initial cup','diamond'],['initial_bowl','Initial bowl','diamond-open']]){let points=rows.map(r=>r[key]);traces.push({type:'scatter3d',mode:'markers',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),name:label,marker:{color:'#333',size:3,symbol},hoverinfo:'name'});}
if(rows.length&&!single){let points=rows.flatMap(r=>[...r.tcp,...r.target,r.initial_cup,r.initial_bowl]);let lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];for(const p of points)for(let i=0;i<3;i++){lo[i]=Math.min(lo[i],p[i]-.02);hi[i]=Math.max(hi[i],p[i]+.02)}let x0=Math.max(.29,lo[0]),x1=Math.min(.76,hi[0]),y=Math.min(.39,Math.max(-.15,lo[1]));if(x0<=x1)traces.push({type:'scatter3d',mode:'lines',x:[x0,x1],y:[y,y],z:[.672,.672],name:'Shelf height',line:{color:'#988b7b',dash:'dot',width:3},hoverinfo:'name'});}
let pathScene={xaxis:{title:'Root X (m)'},yaxis:{title:'Root Y (m)'},zaxis:{title:'Root Z (m)'},aspectmode:'data'};
if(single){let points=[...single.tcp,...single.target,single.initial_cup,single.initial_bowl];for(const [axis,k] of [['xaxis',0],['yaxis',1],['zaxis',2]]){let lo=Math.min(...points.map(p=>p[k])),hi=Math.max(...points.map(p=>p[k])),middle=(lo+hi)/2,span=Math.max(.12,hi-lo+.05);pathScene[axis].range=[middle-span/2,middle+span/2];}}
Plotly.react('plot',traces,{margin:{l:0,r:0,t:10,b:0},scene:pathScene,legend:{orientation:'h'},uirevision:single?'case-'+single.case_id:'snapshot'}, {responsive:true,displaylogo:false});}
let timer=null;function stopAnimation(){if(timer!==null)clearTimeout(timer);timer=null;el('path_play').textContent='Play';}
function playNext(){let slider=el('step');if(slider.disabled||Number(slider.value)>=Number(slider.max)){stopAnimation();return;}let r=DATA.cases.find(x=>String(x.case_id)===el('case').value);let i=Number(slider.value);let delay=1000*(r.step[i+1]-r.step[i])/20/Number(el('path_speed').value);timer=setTimeout(()=>{slider.value=i+1;render();playNext()},delay);}
el('path_play').addEventListener('click',()=>{if(timer!==null){stopAnimation();return;}if(Number(el('step').value)>=Number(el('step').max))el('step').value=0;el('path_play').textContent='Pause';playNext()});
function showCasePath(id){stopAnimation();for(const k of ['object','template','outcome'])el(k).value='all';el('case').value=String(id);el('view').value='paths';render();el('plot').scrollIntoView({behavior:'smooth',block:'center'});}
for(const id of ['view','density_kind','object','template','outcome','case','tcp','target','cabinet_context'])el(id).addEventListener('change',()=>{stopAnimation();render()});el('step').addEventListener('input',()=>{stopAnimation();render()});let initialCase=DATA.cases.find(r=>r.case_id===2)||DATA.cases.find(r=>r.metrics.calibration_success)||DATA.cases[0];if(initialCase){el('case').value=String(initialCase.case_id);el('view').value='paths'}render();</script></body></html>'''


def replay_panel(directory):
    """Embed verified representative movies so the page remains usable offline."""
    if directory is None:
        return ''
    from html import escape
    manifest = json.loads((directory/'replay_manifest.json').read_text())
    clips = {}
    for case in manifest['cases']:
        video = base64.b64encode((directory/case['video']).read_bytes()).decode('ascii')
        label = f"Case {case['case_id']} · {case['object']} · {case['template']}"
        result = 'Complete pipeline' if case['success'] else str(case['failure'])
        clips.setdefault(case['object'], []).append(f'<article style="flex:1;min-width:300px"><h3>{escape(label)}</h3>'
                     f'<p>{escape(result)}</p><button type="button" onclick="showCasePath({int(case["case_id"])})">Show complete 3D trajectory</button><video controls playsinline preload="metadata" '
                     f'style="width:100%;background:#222" aria-label="{escape(label)}">'
                     f'<source src="data:video/mp4;base64,{video}" type="video/mp4"></video></article>')
    return ('<section id="robot_replays"><h2>Watch what the robot actually did</h2>'
            '<p>Verified action replay of selected representative recorded cases. Press play or scrub the video. '
            'The overlay shows the action phase, confirmed grasp feedback and target height change. '
            'These examples do not represent every case in the distribution below.</p>'
            '<label>Playback speed <select id="replay_speed"><option value="0.25">0.25×</option>'
            '<option value="0.5">0.5×</option><option value="1" selected>1× (real time)</option>'
            '<option value="2">2×</option></select></label>'
            + ''.join(f'<h2>{escape(obj.title())}</h2><div style="display:flex;gap:16px;flex-wrap:wrap">'
                      + ''.join(group) + '</div>' for obj, group in clips.items())
            + '<script>document.getElementById("replay_speed").onchange=function(){'
            'document.querySelectorAll("#robot_replays video").forEach(v=>v.playbackRate=Number(this.value));'
            '};</script></section>')


def write_html(path, bundle, records, metadata, replays=None):
    # The bundle is already an inline script; escape a closing tag if one appears.
    import re
    safe_bundle = re.sub(r'</script', r'<\\/script', bundle, flags=re.IGNORECASE)
    html = (HTML.replace('__TIME__', metadata['captured_at_london'])
            .replace('__PLOTLY__', safe_bundle)
            .replace('__VOXEL_SIZE__', str(metadata['density']['cell_size_m']))
            .replace('__DATA__', json_script(dict(cases=[display_case(r, max_points=len(r['tcp'])) for r in records], unfinished=metadata['unfinished'], voxel_size=metadata['density']['cell_size_m'])))
            .replace('__COLORS__', json_script(COLORS)).replace('__LABELS__', json_script(LABELS))
            .replace('__TEMPLATES__', json_script(TEMPLATES)))
    html = html.replace('<div class="controls">', replay_panel(replays)+'<div class="controls">', 1)
    path.write_text(html, encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plotly-source', type=Path, default=DEFAULT_PLOTLY)
    parser.add_argument('--voxel-size', type=float, default=.02)
    parser.add_argument('--replays', type=Path, help='Verified robot movies from render_cabinet_grasp_replays')
    args = parser.parse_args(argv)
    if not np.isfinite(args.voxel_size) or args.voxel_size <= 0:
        raise ValueError('Voxel size must be finite and positive')
    if args.output.exists(): raise FileExistsError(args.output)
    bundle = plotly_bundle(args.plotly_source)
    manifest, closed = freeze_closed(args.input)
    captured = datetime.now(ZoneInfo('Europe/London')).isoformat(timespec='seconds')
    records = [read_case(case, path) for case, path in closed]
    for record in records:
        add_voxels(record, args.voxel_size)
    density = dict(definition='Recorded-position visitation; distinct cases per cell; each case counted once; gaps between samples not filled',
                   includes_initial=True, coordinates='controller_root_m', origin_m=[0., 0., 0.], cell_size_m=args.voxel_size,
                   markers='cell centres, not filled volumes',
                   total_visited_cells={kind: len(voxel_visits(records, kind)) for kind in ('tcp', 'target')},
                   total_case_visits={kind: sum(voxel_visits(records, kind).values()) for kind in ('tcp', 'target')},
                   max_cases_per_cell={kind: max(voxel_visits(records, kind).values(), default=0) for kind in ('tcp', 'target')})
    counts = {outcome: sum(r['outcome'] == outcome for r in records) for outcome in OUTCOMES}
    metadata = dict(captured_at_london=captured, input=str(args.input.resolve()),
                    plotly_source=str(args.plotly_source.resolve()), planned=len(manifest['cases']),
                    closed=len(records), unfinished=len(manifest['cases'])-len(records), outcome_counts=counts,
                    case_ids=[r['case_id'] for r in records], raw_ticks=sum(r['raw_ticks'] for r in records),
                    display_max_points_per_case=None, display_sampling='All recorded controller steps, including reset', density=density, cases=[{k:v for k,v in r.items() if k not in ('tcp','target','initial_cup','initial_bowl','tcp_voxels','target_voxels','step','phase','target_grasped','stable_grasp_ticks',
                     'closed_command','confirmed_grasp','target_height_relative_initial_m')} for r in records])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'snapshot.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    write_html(args.output/'trajectories.html', bundle, records, metadata, args.replays)
    overview(args.output/'overview.png', records)
    density_figure(args.output/'density.png', records, args.voxel_size)
    print(json.dumps({k:metadata[k] for k in ('closed', 'unfinished', 'raw_ticks', 'outcome_counts')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
