"""Pipeline de entrenamiento offline de AgriVision (se ejecuta una sola vez).

Orquesta todas las fases del prompt en orden:
  Inspección (Paso 1) → Verificaciones (Paso 5) → HPO (Optuna) →
  CV estratificada 5-fold → Modelo final → Test (una sola vez) →
  Calibración + umbrales por clase → Guardado del modelo + 5 gráficos.

Uso:
  python scripts/01_train.py                      # pipeline completo
  python scripts/01_train.py --trials 40
  python scripts/01_train.py --smoke               # prueba rápida (subset)
  python scripts/01_train.py --limit-per-class 60 --trials 4 --epochs 3 --workers 0
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "config"))

import config as cfg  # noqa: E402
from src.data import inspection, verification  # noqa: E402
from src.data.dataset import LeafDataset, build_loader, scan_split, subset  # noqa: E402
from src.data.transforms import build_train_transforms, build_eval_transforms  # noqa: E402
from src.evaluation import metrics as M, visualization as viz  # noqa: E402
from src.models.architecture import build_model, get_gradcam_layer, trainable_summary  # noqa: E402
from src.training.calibration import calibrate, calibrated_probs  # noqa: E402
from src.training.cross_validation import run_cross_validation  # noqa: E402
from src.training.hpo import run_hpo  # noqa: E402
from src.training.trainer import evaluate, train_model  # noqa: E402
from src.inference.predict import save_checkpoint  # noqa: E402
from src.utils.helpers import device_info, get_device, make_amp, save_json, set_seed, to_py  # noqa: E402

from sklearn.model_selection import train_test_split  # noqa: E402


def _limit_per_class(paths, labels, n):
    """Submuestrea n imágenes por clase (para pruebas rápidas)."""
    paths = np.asarray(paths); labels = np.asarray(labels)
    keep = []
    for c in range(len(cfg.CLASSES)):
        idx = np.where(labels == c)[0][:n]
        keep.extend(idx.tolist())
    keep = np.asarray(sorted(keep))
    return paths[keep].tolist(), labels[keep]


def main():
    ap = argparse.ArgumentParser(description="Entrenamiento de AgriVision")
    ap.add_argument("--trials", type=int, default=cfg.N_OPTUNA_TRIALS)
    ap.add_argument("--epochs", type=int, default=None, help="override de épocas (todas las fases)")
    ap.add_argument("--workers", type=int, default=cfg.NUM_WORKERS)
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--skip-inspection", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="prueba rápida end-to-end")
    args = ap.parse_args()

    if args.smoke:
        args.limit_per_class = args.limit_per_class or 40
        args.trials = min(args.trials, 3)
        args.epochs = args.epochs or 2
        args.workers = 0

    cfg.NUM_WORKERS = args.workers
    if args.epochs is not None:
        cfg.HPO_EPOCHS = cfg.CV_EPOCHS = cfg.FINAL_EPOCHS = args.epochs

    set_seed(cfg.RANDOM_SEED)
    device = get_device()
    version_id = cfg.make_version_id(date.today().isoformat())
    print(f"[AgriVision] versión={version_id}")
    print(f"[device] {device_info(device)}")

    # ----- Datos -----
    train_paths, train_labels = scan_split(cfg.TRAIN_VALID_DIR)
    test_paths, test_labels = scan_split(cfg.TEST_DIR)
    if args.limit_per_class:
        train_paths, train_labels = _limit_per_class(train_paths, train_labels, args.limit_per_class)
        test_paths, test_labels = _limit_per_class(test_paths, test_labels, max(20, args.limit_per_class // 2))
    print(f"[datos] train={len(train_paths)} | test={len(test_paths)}")

    summary: dict = {"version_modelo": version_id, "backbone": cfg.BACKBONE}

    # ----- Paso 1: inspección/limpieza -----
    if not args.skip_inspection:
        print("[paso 1] inspeccionando dataset...")
        report = inspection.inspect_dataset(train_paths, train_labels)
        save_json(report, cfg.ARTIFACTS_DIR / f"{version_id}_inspeccion.json")
        print(f"  baja_res={report['resumen']['n_baja_resolucion']} "
              f"| exceso_fondo={report['resumen']['n_exceso_fondo']} "
              f"| fuga_potencial={report['resumen']['n_fuga_potencial']}")
        summary["inspeccion"] = report["resumen"]

    # ----- Paso 5: verificaciones de consistencia -----
    print("[paso 5] verificaciones de consistencia...")
    check_ds = LeafDataset(train_paths, train_labels, transform=build_train_transforms())
    check_loader = build_loader(check_ds, 16, device, shuffle=True)
    grid_path = verification.save_augmented_batch_grid(
        check_loader, cfg.RESULTADOS_DIR / f"{version_id}_batch_aumentado.png")
    eval_tf = build_eval_transforms()
    det_ok = verification.eval_transform_is_deterministic(eval_tf)
    norm_stats = verification.normalization_stats(check_loader)
    print(f"  batch aumentado guardado | val_determinista={det_ok} "
          f"| normalizacion_ok={norm_stats['ok']}")
    summary["verificaciones"] = {"val_determinista": det_ok, "normalizacion": norm_stats,
                                 "grid": grid_path}

    # ----- HPO con Optuna -----
    print(f"[hpo] optimizando {args.trials} trials (TPE + pruning)...")
    storage = f"sqlite:///{(cfg.OPTUNA_DIR / f'{version_id}.db').as_posix()}"
    cfg.OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    hpo = run_hpo(train_paths, train_labels, device, n_trials=args.trials,
                  study_name=version_id, storage=storage)
    best = hpo["best_params"]
    print(f"  mejores hiperparámetros: {best}")
    print(f"  best MCC={hpo['best_value']:.4f} | pruned={hpo['n_pruned']}/{hpo['n_trials']}")
    summary["hpo"] = {"best_params": best, "best_value": hpo["best_value"],
                      "n_trials": hpo["n_trials"], "n_pruned": hpo["n_pruned"]}

    # ----- Validación cruzada 5-fold -----
    print(f"[cv] validación cruzada {cfg.N_FOLDS}-fold...")
    cv = run_cross_validation(train_paths, train_labels, best, device, verbose=True)
    # Imágenes por clase en cada fold de validación = (total por clase) / n_folds.
    per_class_total = int(np.bincount(train_labels).min())
    expected_per_class = per_class_total // cfg.N_FOLDS
    balance = verification.verify_fold_balance(
        train_labels, cv["val_indices"], expected_per_class=expected_per_class)
    agg = cv["agregado"]
    print(f"  MCC={agg['mcc']['media']:.3f}±{agg['mcc']['std']:.3f} "
          f"| recall_tardio={agg['recall']['tizon_tardio']['media']:.3f}"
          f"±{agg['recall']['tizon_tardio']['std']:.3f} | balanceado={balance['balanceado']}")
    summary["cv"] = {"agregado": agg, "balance": balance["balanceado"]}

    # Métrica de entrenamiento por clase (media CV de la métrica principal).
    train_metrics_per_class = {}
    for c in cfg.CLASSES:
        principal = cfg.CLASS_METRIC_TARGETS[c]["metrica_principal"]
        key = "recall" if principal == "recall" else "precision"
        train_metrics_per_class[c] = agg[key][c]["media"]

    # ----- Modelo final (holdout estratificado para early-stop + calibración) -----
    print("[final] entrenando modelo final...")
    idx = np.arange(len(train_labels))
    tr_idx, cal_idx = train_test_split(idx, test_size=0.10, stratify=train_labels,
                                       random_state=cfg.RANDOM_SEED)
    tr_p, tr_l = subset(train_paths, train_labels, tr_idx)
    cal_p, cal_l = subset(train_paths, train_labels, cal_idx)

    final_train_ds = LeafDataset(tr_p, tr_l, transform=build_train_transforms())
    cal_ds = LeafDataset(cal_p, cal_l, transform=build_eval_transforms())
    final_train_loader = build_loader(final_train_ds, best["batch_size"], device, shuffle=True)
    cal_loader = build_loader(cal_ds, best["batch_size"], device, shuffle=False)

    model = build_model(dropout=best["dropout"], num_trainable_blocks=best["num_trainable_blocks"])
    print(f"  {trainable_summary(model)}")
    final = train_model(model, final_train_loader, cal_loader, device=device,
                        lr=best["lr"], weight_decay=best["weight_decay"],
                        epochs=cfg.FINAL_EPOCHS, patience=cfg.EARLY_STOP_PATIENCE,
                        monitor=cfg.MONITOR_METRIC, verbose=True)

    # ----- Calibración (temperatura + umbrales por clase) sobre el holdout -----
    print("[calibración] temperature scaling + umbrales por clase...")
    cal_eval = evaluate(model, cal_loader, device, make_amp(device)[1])
    calib = calibrate(cal_eval["logits"], cal_eval["y_true"])
    print(f"  T={calib['temperatura']:.3f} | umbrales={calib['umbrales_por_clase']}")

    # ----- Test (UNA sola vez) -----
    print("[test] evaluación final (una sola vez)...")
    test_ds = LeafDataset(test_paths, test_labels, transform=build_eval_transforms())
    test_loader = build_loader(test_ds, best["batch_size"], device, shuffle=False)
    test_eval = evaluate(model, test_loader, device, make_amp(device)[1])
    # Predicciones con probabilidades calibradas.
    test_probs = calibrated_probs(test_eval["logits"], calib["temperatura"])
    y_pred_cal = test_probs.argmax(axis=1)
    test_report = M.full_report(test_eval["y_true"], y_pred_cal)
    print(f"  TEST MCC={test_report['global']['mcc']:.3f} "
          f"| recall_tardio={test_report['por_clase']['tizon_tardio']['recall']:.3f} "
          f"| listo_produccion={test_report['produccion']['listo_produccion']}")
    summary["test"] = test_report

    # ----- Guardado del modelo + metadatos -----
    _, layer_name = get_gradcam_layer(model)
    metadata = {
        "version_modelo": version_id,
        "fecha": date.today().isoformat(),
        "backbone": cfg.BACKBONE,
        "hiperparametros": best,
        "capa_gradcam": layer_name,
        "temperatura": calib["temperatura"],
        "umbrales_por_clase": calib["umbrales_por_clase"],
        "umbrales_detalle": calib["umbrales_detalle"],
        "metricas_entrenamiento_por_clase": train_metrics_per_class,
        "metricas_test": test_report,
        "clases": cfg.CLASSES,
    }
    ckpt_path = cfg.MODELS_DIR / f"{version_id}.pt"
    save_checkpoint(ckpt_path, model, metadata)
    save_json(to_py(metadata), cfg.MODELS_DIR / f"{version_id}_metadata.json")
    print(f"[guardado] modelo -> {ckpt_path}")

    # ----- Los 5 gráficos -----
    print("[gráficos] generando los 5 .png...")
    paths = {
        "curvas_entrenamiento": viz.plot_training_curves(final["history"], version_id),
        "matriz_confusion": viz.plot_confusion_matrix(
            M.confusion(test_eval["y_true"], y_pred_cal, normalize="true"), version_id),
        "sensibilidades": viz.plot_sensitivities(test_report["por_clase"], version_id),
        "boxplot_cv": viz.plot_cv_boxplot(cv["por_fold"], version_id),
        "curvas_roc": viz.plot_roc_curves(calib["roc"], version_id),
    }
    for name, p in paths.items():
        print(f"  {name}: {p}")

    summary["graficos"] = paths
    save_json(to_py(summary), cfg.ARTIFACTS_DIR / f"{version_id}_resumen.json")
    print(f"[fin] resumen -> {cfg.ARTIFACTS_DIR / f'{version_id}_resumen.json'}")


if __name__ == "__main__":
    main()
