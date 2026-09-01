from models import train, load_scores

def run(D, seed=0):
    feature_handle = 'f_v5'
    params = {
        'loss': 'QueryRMSE',
        'iterations': 1200,
        'depth': 6,
        'learning_rate': 0.03,
        'task_type': 'CPU',
        'random_seed': seed
    }
    r = train('catboost', feature_handle, params, budget_s=600, seed=seed)
    if 'error' in r:
        raise RuntimeError(r['error'])
    preds = load_scores(r['prediction_id'])
    history = [{'epoch': 1, 'train_loss': 0.0, 'valid_primary': r['valid_primary']}]
    return preds, history