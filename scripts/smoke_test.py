"""Exercise the actual example workflow without changing existing run records."""
from pathlib import Path
import json
import os
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import engine


def main():
    original_out=engine.OUT
    original_cwd=Path.cwd()
    with tempfile.TemporaryDirectory(prefix='lai-smoke-') as temp:
        engine.OUT=Path(temp)/'outputs'
        engine.OUT.mkdir()
        # Start outside the project to verify both form and CSV-relative paths.
        os.chdir(temp)
        import app as web
        client=web.app.test_client()
        headers={'X-LAI-Token':web.TOKEN}

        def run(params, expected='completed'):
            response=client.post('/api/run',json=params,headers=headers)
            assert response.status_code==202,response.get_data(as_text=True)
            job_id=response.get_json()['id']
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                job=client.get('/api/jobs/'+job_id).get_json()
                # Status becomes final slightly before finally writes run.json.
                if job['status'] in ('completed','failed') and 'elapsed_seconds' in job:
                    break
                time.sleep(.03)
            else:raise TimeoutError(job_id)
            assert job['status']==expected,job
            print(params['action'],job['status'],job['elapsed_seconds'],'s')
            if job.get('preview'):
                response=client.get('/files/'+job_id+'/'+job['preview'])
                assert response.status_code==200 and len(response.data)>100
            return job

        try:
            response=client.get('/')
            assert response.status_code==200 and 'examples/example_features.csv' in response.get_data(as_text=True)
            table=client.get('/api/table?path=examples/example_features.csv').get_json()
            assert table['rows']==24 and len(table['dates'])==2
            assert set(engine.FEATURES)<=set(table['columns'])
            assert client.post('/api/run',json={'action':'train'}).status_code==403
            assert client.post('/api/run',json=[],headers=headers).status_code==400

            extracted=run(dict(action='extract',samples='examples/sample_locations.csv',
                               imagery_config='examples/imagery_config.csv',sample_crs='EPSG:32649',q=70,p=90))
            assert extracted['observations']==24 and extracted['valid']==24
            feature_path=engine.OUT/extracted['id']/'extracted_features.csv'
            trained=run(dict(action='train',table=str(feature_path),pipeline='public_reference_v1',
                             n_estimators=120,learning_rate=.03,max_depth=2))
            assert trained['metrics']['tree_ids']==12 and trained['metrics']['observations']==24
            meta=json.loads((engine.OUT/trained['id']/'model_meta.json').read_text(encoding='utf-8'))
            assert meta['parameters']['n_estimators']==120 and meta['parameters']['max_depth']==2
            predicted=run(dict(action='predict',table='examples/example_prediction.csv',
                               model_id=trained['id'],pipeline='public_reference_v1'))
            assert predicted['valid']==12 and predicted['invalid']==0
            csv=engine.read_table(engine.OUT/predicted['id']/'lai_predictions.csv')
            assert csv.LAI_Predicted.notna().all() and len(csv)==12
            assert client.get('/files/'+predicted['id']+'/lai_predictions.csv?download=1').status_code==200
            mapped=run(dict(action='map',map_mode='raster',raster='examples/lai_demo.tif'))
            assert mapped['metadata']['width']==128 and mapped['metadata']['crs']=='EPSG:32649'
            shown=run(dict(action='map',map_mode='figure',figure='examples/lai_demo.png'))
            assert (engine.OUT/shown['id']/shown['preview']).read_bytes()==(ROOT/'examples/lai_demo.png').read_bytes()
            changed=run(dict(action='extract',samples='examples/sample_locations.csv',
                             imagery_config='examples/imagery_config.csv',sample_crs='EPSG:32649',
                             q=60,p=85,exploration_radius_m=2.1,alpha=.95,min_valid_pixels=20))
            assert changed['valid']==24 and changed['roi_parameters']['q_percentile']==60
            failed=run(dict(action='extract',samples='examples/sample_locations.csv',
                            imagery_config='examples/imagery_config.csv',sample_crs='EPSG:32649',q=100),expected='failed')
            assert 'q' in failed['message']
            assert len(client.get('/api/jobs').get_json())==7
            print('PASS: example inputs, 4 functions, parameter changes, downloads, path portability.')
        finally:
            web.executor.shutdown(wait=True)
            web.OUT=original_out
            engine.OUT=original_out
            os.chdir(original_cwd)


if __name__=='__main__':
    main()
