# fl_judge.py
"""
Módulo simple para decidir parada en FL usando:
  - max_rounds (tope absoluto)
  - early stopping sobre val_acc global (con patience y min_delta)

Uso:
    from fl_judge import Judge
    judge = Judge(max_rounds=50, patience=5, min_delta=1e-4)
    continue_training = judge.update_and_should_continue(round_idx, val_acc)
"""

from typing import Optional
import logging

class Judge:
    def __init__(self, max_rounds: int = 50, patience: int = 5, min_delta: float = 1e-6):
        """
        max_rounds: número máximo de rondas globales (inclusive).
        patience: número de rondas consecutivas sin mejora de val_acc antes de parar.
        min_delta: mínima mejora en val_acc para considerarla real (evita ruido).
        """
        self.max_rounds = int(max_rounds)
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        # estado interno
        self.best_val: float = -float("inf")
        self.no_improve_count: int = 0

    def reset(self):
        """Resetear estado interno."""
        self.best_val = -float("inf")
        self.no_improve_count = 0

    def update_and_should_continue(self, round_idx: int, val_acc: Optional[float]) -> bool:
        """
        Debe llamarse **después** de evaluar el modelo global (y obtener val_acc).
        - round_idx: entero (ronda actual, por ejemplo 1,2,...).
        - val_acc: accuracy de validación global (float en [0,1]). Si es None, solo se aplica max_rounds.

        Devuelve True si **DEBE CONTINUAR**, False si **DEBE PARAR**.
        """
        # 1) tope absoluto
        if round_idx >= self.max_rounds:
            logging.info(f"[Judge] Parar: round {round_idx} >= max_rounds {self.max_rounds}")
            return False

        # 2) si val_acc no está disponible, no podemos hacer early stopping -> continuar
        if val_acc is None:
            logging.debug("[Judge] val_acc no proporcionado; usar solo max_rounds.")
            return True

        # 3) early stopping: comparar con la mejor val vista
        if val_acc > self.best_val + self.min_delta:
            logging.debug(f"[Judge] Mejora val: {self.best_val:.6f} -> {val_acc:.6f}")
            self.best_val = val_acc
            self.no_improve_count = 0
            return True
        else:
            self.no_improve_count += 1
            logging.debug(f"[Judge] No mejora: contador {self.no_improve_count}/{self.patience}")
            if self.no_improve_count >= self.patience:
                logging.info(f"[Judge] Parar por early stopping: {self.no_improve_count} rondas sin mejora.")
                return False
            return True
