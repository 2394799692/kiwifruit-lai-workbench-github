"""Run python app.py, then open http://127.0.0.1:8765 in a local browser."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json, secrets, time, traceback, threading, webbrowser, argparse
from flask import Flask, jsonify, render_template, request, send_from_directory, abort
from engine import ROOT, OUT, ACTIONS, read_table, write_json

app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=2*1024*1024
app.config['TEMPLATES_AUTO_RELOAD']=True
TOKEN=secrets.token_urlsafe(24);executor=ThreadPoolExecutor(max_workers=1);lock=threading.Lock();jobs={}
CONFIG=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))

@app.before_request
def local_access():
    if request.host.split(':')[0] not in ['127.0.0.1','localhost']:abort(403)
    if request.method=='POST' and request.headers.get('X-LAI-Token')!=TOKEN:abort(403)

@app.get('/')
def index():return render_template('index.html',token=TOKEN,config=CONFIG)

def load_job(job_id):
    if Path(job_id).name!=job_id:abort(404)
    if job_id in jobs:return jobs[job_id]
    path=OUT/job_id/'run.json'
    if not path.exists():abort(404)
    return json.loads(path.read_text(encoding='utf-8'))

@app.get('/api/jobs')
def list_jobs():
    result=[]
    for p in sorted(OUT.glob('*/run.json'),reverse=True):
        j=json.loads(p.read_text(encoding='utf-8'));result.append(j)
    return jsonify(result[:30])

@app.get('/api/jobs/<job_id>')
def get_job(job_id):return jsonify(load_job(job_id))

@app.get('/files/<job_id>/<path:filename>')
def files(job_id,filename):
    load_job(job_id)
    folder=(OUT/job_id).resolve();target=(folder/filename).resolve()
    if folder not in target.parents or target.suffix in ['.joblib','.py']:abort(404)
    return send_from_directory(folder,filename,as_attachment=request.args.get('download')=='1')

@app.post('/api/run')
def run():
    params=request.get_json(force=True)
    if not isinstance(params,dict):return jsonify(error='请求须为 JSON 对象。'),400
    action=params.get('action')
    if action not in ACTIONS:return jsonify(error='未知操作'),400
    with lock:
        if any(j['status'] in ['queued','running'] for j in jobs.values()):return jsonify(error='已有任务运行中，请等待完成。'),409
        job_id=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(3)
        dest=OUT/job_id;dest.mkdir()
        j={'id':job_id,'action':action,'status':'queued','started':datetime.now().isoformat(timespec='seconds'),'message':'等待运行','output_dir':str(dest),'parameters':params}
        jobs[job_id]=j;write_json(dest/'run.json',j)
    def work():
        start=time.perf_counter()
        def progress(s):j['message']=s;write_json(dest/'run.json',j)
        try:
            j['status']='running';progress('正在检查输入数据。')
            result=ACTIONS[action](params,dest,progress);j.update(result);j['status']='completed'
        except Exception as e:
            j['status']='failed';j['message']=str(e);(dest/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        finally:
            j['elapsed_seconds']=round(time.perf_counter()-start,2)
            j['files']=[p.name for p in dest.iterdir() if p.is_file() and p.suffix!='.joblib']
            write_json(dest/'run.json',j)
    executor.submit(work)
    return jsonify(id=job_id),202

@app.get('/api/table')
def table():
    try:
        df=read_table(request.args.get('path',CONFIG['demo_features']))
        values=json.loads(df.head(20).to_json(orient='records',force_ascii=False))
        return jsonify(rows=len(df),columns=df.columns.tolist(),dates=df.Date.astype(str).unique().tolist() if 'Date' in df else [],preview=values)
    except Exception as e:return jsonify(error=str(e)),400

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--no-browser',action='store_true');parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    if not 1<=args.port<=65535:parser.error('port 必须在 1～65535 之间')
    url=f'http://127.0.0.1:{args.port}'
    if not args.no_browser:threading.Timer(1.2,lambda:webbrowser.open(url)).start()
    print(f'猕图知势 LAI 工作台：{url}  |  Ctrl+C 结束')
    app.run(host='127.0.0.1',port=args.port,debug=False,threaded=True)
