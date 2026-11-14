"""Offline browser validation of all arms and reset-safe exact trajectories."""
from pathlib import Path
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
import websocket

OUT = Path(__file__).resolve().parent
expected = json.loads((OUT/'summary.json').read_text())
with tempfile.TemporaryDirectory(prefix='pick-place-browser-', dir='/tmp', ignore_cleanup_errors=True) as profile:
    process = subprocess.Popen(['/usr/bin/google-chrome','--headless','--no-sandbox','--disable-dev-shm-usage','--remote-debugging-port=0','--remote-allow-origins=*','--user-data-dir='+profile,'about:blank'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ws = None
    try:
        active = Path(profile)/'DevToolsActivePort'
        for _ in range(100):
            if active.exists(): break
            time.sleep(.1)
        port = active.read_text().splitlines()[0]
        pages = json.load(urllib.request.urlopen('http://127.0.0.1:'+port+'/json'))
        ws = websocket.create_connection(pages[0]['webSocketDebuggerUrl'], timeout=60)
        serial = 0
        errors = []
        def call(method, params=None):
            global serial
            serial += 1
            ident = serial
            ws.send(json.dumps(dict(id=ident,method=method,params=params or {})))
            while True:
                result = json.loads(ws.recv())
                if result.get('method') == 'Runtime.exceptionThrown': errors.append(result['params'])
                if result.get('id') == ident:
                    assert 'error' not in result, result
                    return result.get('result',{})
        def evaluate(expression):
            result = call('Runtime.evaluate',dict(expression=expression,awaitPromise=True,returnByValue=True))
            assert 'exceptionDetails' not in result, result
            return result['result'].get('value')
        call('Runtime.enable')
        call('Page.enable')
        call('Emulation.setDeviceMetricsOverride',dict(width=1550,height=1180,deviceScaleFactor=1,mobile=False))
        call('Page.navigate',dict(url=(OUT/'trajectories.html').as_uri()))
        for _ in range(150):
            if evaluate('!!window.trajectoryViewer && window.trajectoryViewer.state.arm === "random" && !!document.getElementById("pathChart").data'): break
            time.sleep(.1)
        audit = dict(arms={})
        for arm in ['random','curiosity_heuristic','curiosity_only','heuristic_only']:
            evaluate('document.getElementById("arm").value='+json.dumps(arm))
            evaluate('window.trajectoryViewer.loadArm('+json.dumps(arm)+')')
            evaluate('window.trajectoryViewer.renderCharts()')
            result = evaluate('''(()=>{const v=window.trajectoryViewer,s=v.state;const seg=v.makeSegments();return {arm:s.arm,n:s.n,skills:s.skills.length,error:document.getElementById('error').textContent,plotCount:document.querySelectorAll('.js-plotly-plot').length,resetSafe:seg.every(g=>g.indices.every(i=>i===0||s.episodes[i-1]===s.episodes[g.indices[g.indices.length-1]-1])),details:document.getElementById('details').textContent}})()''')
            assert result['n'] == expected[arm]['actual_controls'], result
            assert result['skills'] == expected[arm]['recorded_macro_ranges'], result
            assert result['plotCount'] == 2 and result['resetSafe'] and not result['error'], result
            audit['arms'][arm] = result
            # First real episode after reset: verify exact next-state endpoints.
            result = evaluate('''(async()=>{const v=window.trajectoryViewer,s=v.state;document.getElementById('episode').value=String(s.episodeIds[1]);s.selectedSkill=null;await v.renderCharts();let seg=v.makeSegments()[0];return {count:seg.indices.length,expected:s.episodes.filter(e=>e===s.episodeIds[1]).length,resetSafe:seg.indices.every(i=>i>0&&s.episodes[i-1]===s.episodeIds[1])}})()''')
            assert result['count'] == result['expected'] and result['resetSafe'], result
            audit['arms'][arm]['exact_episode'] = result
        evaluate('document.getElementById("arm").value="random";window.trajectoryViewer.loadArm("random")')
        result = evaluate('''(async()=>{const v=window.trajectoryViewer,s=v.state,sk=s.skills.find(x=>x.status==='success');v.selectSkill(sk);await v.renderCharts();document.getElementById('statusFilter').value='success';document.getElementById('episode').value='all';v.render();await v.renderCharts();return {successCount:v.filteredSkills().length,selected:sk,points:v.makeSegments()[0].indices.length,details:document.getElementById('details').textContent}})()''')
        assert result['successCount'] == 3 and result['selected']['kind'] == 'grasp', result
        assert result['points'] == result['selected']['end']-result['selected']['start']+1, result
        audit['successful_grasp'] = result
        image = call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))['data']
        (OUT/'successful_grasp.png').write_bytes(base64.b64decode(image))
        evaluate('document.getElementById("episode").value="all";window.trajectoryViewer.state.selectedSkill=null;window.trajectoryViewer.renderCharts()')
        audit['overview'] = evaluate('''(()=>{const v=window.trajectoryViewer,s=v.state,g=v.makeSegments();return {segments:g.length,expected:s.episodeIds.length,resetSafe:g.every(seg=>seg.indices.every(i=>i===0||s.episodes[i-1]===s.episodes[seg.indices[seg.indices.length-1]-1]))}})()''')
        assert audit['overview']['segments'] == audit['overview']['expected'] and audit['overview']['resetSafe']
        audit['offline_resources'] = evaluate('performance.getEntriesByType("resource").map(x=>x.name)')
        assert all(url.startswith('file:') for url in audit['offline_resources'])
        assert not errors, errors
        audit['uncaught_exceptions'] = errors
        (OUT/'browser_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(json.dumps(audit,indent=2))
    finally:
        if ws: ws.close()
        process.terminate()
        process.wait(timeout=15)
