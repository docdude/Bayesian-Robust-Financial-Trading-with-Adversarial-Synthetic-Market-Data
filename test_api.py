import sys, os
sys.path.insert(0, 'generator/GRT_GAN')
from models.API import GeneratorAPI

obs_features = ['open','high','low','close','adj_close','kmid','kmid2','klen','kup','kup2','klow','klow2','ksft','ksft2']
temporal_features = ['day','weekday','month']

try:
    gen = GeneratorAPI(
        model_path='generator/GRT_GAN/output/etf_18_120',
        ticker_name='DBB',
        obs_features=obs_features,
        temporal_features=temporal_features
    )
    print('SUCCESS: GeneratorAPI loaded')
    print(f'  output_data shape: {gen.output_data.shape}')
    print(f'  ticker_list: {gen.ticker_list}')
    print(f'  time range: {gen.time[0]} to {gen.time[-1]}')
    print(f'  model device: {gen.args.device}')
except Exception as e:
    print(f'FAILED: {e}')
    import traceback
    traceback.print_exc()
