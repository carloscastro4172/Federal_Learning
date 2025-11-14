# fl_client_with_plots.py
"""
Cliente FL simplificado que:
 - guarda métricas por dataset en data/models/metrics_{dataset}.csv
 - usa Judge (max_rounds + early stopping global)
 - al finalizar guarda UNA IMAGEN PNG con dos subplots:
     izq: Val Accuracy por round (local vs global)
     der: Test Accuracy por round (local vs global)
 - no envía las imágenes al servidor (quedan locales)
"""
import logging
import os
import time
import csv
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from .conversion import Converter
from .tnn_training import DataManager
from .cnn import MLP
from fl_main.agent.client import Client
from .judge import Judge

# ---------------- Config ----------------
DATASET_TAG = os.environ.get("DATASET_TAG", "dataset1")
MODELS_DIR = None  # se inicializa con _models_dir()

# ============ Helpers ============
def _models_dir() -> str:
    global MODELS_DIR
    if MODELS_DIR is None:
        base = os.path.dirname(os.path.abspath(__file__))
        d = os.path.join(base, "data", "models")
        os.makedirs(d, exist_ok=True)
        MODELS_DIR = d
    return MODELS_DIR

def save_models_npz(models: Dict[str, np.ndarray], tag: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(_models_dir(), f"{ts}_{tag}.npz")
    clean = {k.replace("/", "_"): v for k, v in models.items()}
    np.savez_compressed(path, **clean)
    logging.info(f"[checkpoint] {path}")
    return path

def _metrics_csv_path(dataset_tag: str = "dataset1") -> str:
    return os.path.join(_models_dir(), f"metrics_{dataset_tag}.csv")

def log_metrics_csv(round_idx: int, kind: str, val_acc: float, test_acc: float,
                    dataset_tag: str = "dataset1") -> None:
    csv_path = _metrics_csv_path(dataset_tag)
    header = ["timestamp", "round", "kind", "val_acc", "test_acc"]
    row = [time.strftime("%Y-%m-%d %H:%M:%S"), round_idx, kind, val_acc, test_acc]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

# ------------- plotting final (UNA IMAGEN) -------------
def plot_and_save_single_image(dataset_tag: str = "dataset1", save_dir: Optional[str] = None) -> None:
    """
    Lee metrics_{dataset_tag}.csv y guarda UNA IMAGEN PNG que contiene dos subplots:
      - izq: Val Accuracy por round (local vs global)
      - der: Test Accuracy por round (local vs global)
    """
    csv_path = _metrics_csv_path(dataset_tag)
    if not os.path.exists(csv_path):
        logging.warning(f"No existe {csv_path} — no hay métricas para graficar.")
        return

    dfm = pd.read_csv(csv_path, parse_dates=["timestamp"], keep_default_na=True)
    if dfm.empty:
        logging.warning(f"{csv_path} está vacío — nada que graficar.")
        return

    dfm["round"] = dfm["round"].astype(int)
    dfm = dfm.sort_values("round")

    pivot_val = dfm.pivot_table(index="round", columns="kind", values="val_acc")
    pivot_test = dfm.pivot_table(index="round", columns="kind", values="test_acc")

    if save_dir is None:
        save_dir = _models_dir()
    os.makedirs(save_dir, exist_ok=True)

    # Crear figura con 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # VAL subplot (izq)
    ax = axes[0]
    if "local" in pivot_val.columns:
        ax.plot(pivot_val.index, pivot_val["local"], label="local", marker="o", linewidth=2)
    if "global" in pivot_val.columns:
        ax.plot(pivot_val.index, pivot_val["global"], label="global", marker="s", linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Val Accuracy")
    ax.set_title(f"{dataset_tag} — Val Accuracy (local vs global)")
    ax.grid(alpha=0.2)
    ax.legend()

    # TEST subplot (der)
    ax = axes[1]
    if "local" in pivot_test.columns:
        ax.plot(pivot_test.index, pivot_test["local"], label="local", marker="o", linewidth=2)
    if "global" in pivot_test.columns:
        ax.plot(pivot_test.index, pivot_test["global"], label="global", marker="s", linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"{dataset_tag} — Test Accuracy (local vs global)")
    ax.grid(alpha=0.2)
    ax.legend()

    plt.suptitle(f"Resumen de desempeño — {dataset_tag}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = os.path.join(save_dir, f"{dataset_tag}_accuracy_summary.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info(f"[plot] Guardado resumen final: {out_path}")

# ============ Metadatos y funciones de entrenamiento ============
class TrainingMetaData:
    num_training_data = 476

def init_models() -> Dict[str, np.ndarray]:
    dm = DataManager.dm()
    in_dim = dm.input_dim
    conv = Converter.cvtr()
    conv.set_model_ctor(lambda: MLP(in_features=in_dim))
    net = MLP(in_features=in_dim)
    return conv.convert_nn_to_dict_nparray(net)

def training(models: Dict[str, np.ndarray], init_flag: bool = False) -> Dict[str, np.ndarray]:
    conv = Converter.cvtr()
    if init_flag:
        DataManager.dm(int(TrainingMetaData.num_training_data / 4))
        return init_models()

    logging.info("--- Entrenamiento local ---")
    net = conv.convert_dict_nparray_to_nn(models)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    dm = DataManager.dm()
    net.train()
    for epoch in range(10):
        for X_batch, y_batch in dm.trainloader:
            optimizer.zero_grad()
            logits = net(X_batch)
            loss = criterion(logits, y_batch.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
    return conv.convert_nn_to_dict_nparray(net)

def _eval_split(models: Dict[str, np.ndarray], split: str) -> float:
    conv = Converter.cvtr()
    net = conv.convert_dict_nparray_to_nn(models)
    net.eval()

    dm = DataManager.dm()
    loader = dm.valloader if split == "val" else dm.testloader

    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = net(X_batch)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            correct += (preds.squeeze() == y_batch.int()).sum().item()
            total += y_batch.size(0)
    return float(correct) / max(total, 1)

# -------------------------- Main loop --------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Iniciando cliente FL (con Judge + plot final)...")

    fl_client = Client()
    logging.info(f"IP local del agente: {fl_client.agent_ip}")

    # Inicializar judge (ajusta max_rounds/patience/min_delta si quieres)
    judge = Judge(max_rounds=50, patience=5, min_delta=1e-4)

    # Modelo inicial
    initial_models = training(dict(), init_flag=True)
    save_models_npz(initial_models, "init")
    fl_client.send_initial_model(initial_models)

    fl_client.start_fl_client()

    training_count = 0
    gm_arrival_count = 0
    dataset_tag = DATASET_TAG

    while True:
        # Esperar modelo global (tu API probablemente devuelve dict o ruta)
        global_models = fl_client.wait_for_global_model()
        gm_arrival_count += 1
        save_models_npz(global_models, f"global_r{gm_arrival_count}")

        # Evaluar global (val + test) y log
        acc_val_g = _eval_split(global_models, "val")
        acc_tst_g = _eval_split(global_models, "test")
        print(f"[global] Val Acc: {100*acc_val_g:.2f}% | Test Acc: {100*acc_tst_g:.2f}%")
        log_metrics_csv(gm_arrival_count, "global", acc_val_g, acc_tst_g, dataset_tag)

        # Preguntar al juez si continuar (early stopping global + max rounds)
        should_continue = judge.update_and_should_continue(gm_arrival_count, val_acc=acc_val_g)
        if not should_continue:
            logging.info("Criterio de parada satisfecho. Saliendo del bucle principal.")
            break

        # Entrenamiento local
        models = training(global_models)
        training_count += 1
        save_models_npz(models, f"local_r{training_count}")

        # Evaluar local y log
        acc_val_l = _eval_split(models, "val")
        acc_tst_l = _eval_split(models, "test")
        print(f"[local ] Val Acc: {100*acc_val_l:.2f}% | Test Acc: {100*acc_tst_l:.2f}%")
        log_metrics_csv(training_count, "local", acc_val_l, acc_tst_l, dataset_tag)

        # Enviar modelo local
        fl_client.send_trained_model(models, int(TrainingMetaData.num_training_data), acc_tst_l)

    # Al salir del while: generar UNA imagen resumen y guardarla localmente
    plot_and_save_single_image(dataset_tag=dataset_tag)

    logging.info("Cliente finalizado.")
