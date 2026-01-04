import os
import sys
import traceback

# ensure project src is importable
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt

from utils.config import config
from data_processing.data_loader import DataLoader
from data_processing.data_cleaner import DataCleaner
from models.collaborative_filtering import ItemBasedCF
from models.lightgbm_model import LightGBMModel

def try_import_mf():
    try:
        from models.matrix_factorization import MatrixFactorizationModel
        return MatrixFactorizationModel
    except Exception:
        return None

def extract_history(result, model):
    # 优先取 model.train_history
    hist = getattr(model, 'train_history', None)
    if hist:
        return hist
    # 否则尝试从 train 返回值中解析
    if isinstance(result, dict):
        # 常见键
        for k in ('train_history', 'history', 'loss', 'train_loss', 'evals_result'):
            if k in result:
                return result[k]
        # 如果是 evals_result-like {'train':{'rmse':[...],...}}
        for v in result.values():
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, (list,tuple)):
                        return vv
    if isinstance(result, (list,tuple)):
        return list(result)
    return None

def save_plot(name, history, out_dir):
    if not history:
        print(f"[{name}] no history to plot.")
        return None
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6,4))
    plotted = False
    if isinstance(history, dict):
        for k, v in history.items():
            if isinstance(v, (list,tuple)) and len(v)>0:
                plt.plot(v, label=k)
                plotted = True
    elif isinstance(history, (list,tuple)) and len(history)>0:
        plt.plot(history, label='loss')
        plotted = True
    if not plotted:
        plt.close()
        print(f"[{name}] history exists but nothing plottable.")
        return None
    plt.title(f"{name} training")
    plt.xlabel("iter/epoch")
    plt.ylabel("loss / metric")
    plt.legend(fontsize='small')
    out_path = os.path.join(out_dir, f"loss_{name.lower().replace(' ','_')}.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[{name}] saved plot -> {out_path}")
    return out_path

def main():
    print("Start training ensemble submodels and saving loss plots...")
    models_dir = config.get('data.models_path', 'data/models/')
    os.makedirs(models_dir, exist_ok=True)

    loader = DataLoader('data/raw/')
    cleaner = DataCleaner()

    books = loader.load_books_data()
    users = loader.load_users_data()
    ratings = loader.load_ratings_data()

    books = cleaner.clean_books_data(books)
    users = cleaner.clean_users_data(users)
    ratings = cleaner.clean_ratings_data(ratings)
    try:
        ratings = cleaner.filter_sparse_data(ratings)
    except Exception:
        pass

    # instantiate models
    cf = ItemBasedCF()
    lgbm = LightGBMModel()
    mf_cls = try_import_mf()
    mf = mf_cls() if mf_cls else None

    models = [cf, lgbm]
    if mf is not None:
        models.append(mf)

    trained = []
    histories = {}
    for m in models:
        name = getattr(m, 'name', m.__class__.__name__)
        print(f"----> Training {name} ...")
        try:
            # try common signatures
            try:
                res = m.train(ratings, books, users)
            except TypeError:
                try:
                    res = m.train(ratings, books)
                except TypeError:
                    res = m.train(ratings)
        except Exception as e:
            print(f"[{name}] training error: {e}")
            traceback.print_exc()
            res = None

        hist = extract_history(res, m)
        if hist is None:
            # If LightGBM may have stored evals_result in attribute 'evals_result' or similar
            hist = getattr(m, 'train_history', None) or getattr(m, 'evals_result', None)
        histories[name] = hist
        save_plot(name, hist, models_dir)
        trained.append(m)

    # ensemble overview plot
    combined = {}
    for k, v in histories.items():
        if isinstance(v, dict):
            # pick first plottable series
            for kk, vv in v.items():
                if isinstance(vv, (list,tuple)) and vv:
                    combined[f"{k}.{kk}"] = vv
                    break
        elif isinstance(v, (list,tuple)) and v:
            combined[f"{k}.loss"] = v

    if combined:
        plt.figure(figsize=(8,5))
        for k, v in combined.items():
            plt.plot(v, label=k)
        plt.title("Ensemble submodels training overview")
        plt.xlabel("iter/epoch")
        plt.ylabel("loss / metric")
        plt.legend(fontsize='small')
        out_path = os.path.join(models_dir, "loss_ensemble_overview.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"[Ensemble] saved overview -> {out_path}")
    else:
        print("[Ensemble] no plottable history for overview.")

    print("Done.")

if __name__ == "__main__":
    main()