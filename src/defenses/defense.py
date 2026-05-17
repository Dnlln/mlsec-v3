"""
Трёхуровневый конвейер защиты от атак отравления данных (Data Poisoning Defense Pipeline).

Архитектура:
    Уровень 1 — Обнаружение: IsolationForest + LocalOutlierFactor
    Уровень 2 — Согласование: взвешенное голосование (soft/hard)
    Уровень 3 — Адаптация: калибровка под профиль угрозы (backdoor/lf/auto)

Поддерживаемые датасеты: UNSW-NB15, Adult Census Income, SMS Spam Collection, MNIST (бинарный).
Поддерживаемые атаки: Random Label Flipping, Targeted Label Flipping, Backdoor (ConstantPerturbation).

Пример использования:
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> from src.defenses.defense import DefenseTransformer
    >>>
    >>> # Прямое использование
    >>> X_train_clean, y_train_clean = DefenseTransformer(
    ...     contamination=0.10,
    ...     threat_profile='backdoor',
    ...     voting='soft'
    ... ).fit_transform(X_train_poisoned, y_train_poisoned)
    >>>
    >>> # Встраивание в Pipeline sklearn
    >>> pipeline = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('clf', LogisticRegression())
    ... ])
    >>> defense = DefenseTransformer(contamination=0.10)
    >>> X_clean, y_clean = defense.fit_transform(X_train, y_train)
    >>> pipeline.fit(X_clean, y_clean)
"""

import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import check_is_fitted


class DefenseTransformer(BaseEstimator, TransformerMixin):
    """
    Трёхуровневый sklearn-совместимый трансформер защиты от отравления данных.

    Уровень 1 — Обнаружение аномалий двумя независимыми детекторами:
        - IsolationForest: глобальные аномалии (эффективен против backdoor-триггеров).
        - LocalOutlierFactor: локальная плотность (эффективен против кластеров Targeted LF).

    Уровень 2 — Согласование через взвешенное голосование:
        - Итоговый скор: s(x) = alpha * s_IF + (1 - alpha) * s_LOF
        - Режим 'soft': удаление по порогу составного скора.
        - Режим 'hard': удаление только если оба детектора флагируют объект.

    Уровень 3 — Адаптация под профиль угрозы:
        - 'backdoor': повышенная чувствительность IF (contamination * 1.2), сниженная LOF.
        - 'lf': сниженная чувствительность IF (contamination * 0.8), повышенная LOF.
        - 'auto': одинаковые contamination для обоих детекторов.

    Параметры
    ----------
    contamination : float, по умолчанию 0.05
        Ожидаемая доля отравлённых объектов (0 < contamination < 0.5).
        Используется как базовое значение для IF и LOF.
    alpha : float, по умолчанию 0.5
        Вес IsolationForest в составном скоре аномальности.
        (1 - alpha) — вес LocalOutlierFactor.
    voting : {'soft', 'hard'}, по умолчанию 'soft'
        Стратегия голосования:
        - 'soft': удаление по порогу составного скора.
        - 'hard': удаление только если оба детектора независимо флагируют объект.
    threat_profile : {'auto', 'backdoor', 'lf'}, по умолчанию 'auto'
        Профиль угрозы для калибровки агрессивности очистки.
    random_state : int или None, по умолчанию 42
        Seed для воспроизводимости IsolationForest.
    n_jobs : int, по умолчанию 1
        Количество параллельных задач для IsolationForest и LOF.
    if_n_estimators : int, по умолчанию 100
        Количество деревьев в IsolationForest.
    lof_n_neighbors : int, по умолчанию 20
        Количество соседей в LocalOutlierFactor.

    Атрибуты
    ---------
    isolation_forest_ : IsolationForest
        Обученный детектор IsolationForest.
    lof_ : LocalOutlierFactor
        Обученный детектор LocalOutlierFactor.
    contamination_if_ : float
        Фактическое contamination для IsolationForest после применения профиля.
    contamination_lof_ : float
        Фактическое contamination для LOF после применения профиля.
    n_samples_fit_ : int
        Количество объектов в обучающей выборке.

    Примеры
    --------
    >>> import numpy as np
    >>> from src.defenses.defense import DefenseTransformer
    >>> X = np.random.randn(500, 10)
    >>> y = np.random.randint(0, 2, 500)
    >>> # Искусственное отравление
    >>> X[:20] += 10
    >>> y[:20] = 1 - y[:20]
    >>> defense = DefenseTransformer(contamination=0.05, threat_profile='auto')
    >>> X_clean, y_clean = defense.fit_transform(X, y)
    >>> print(f"Удалено объектов: {500 - len(X_clean)}")
    """

    def __init__(
        self,
        contamination: float = 0.05,
        alpha: float = 0.5,
        voting: str = 'soft',
        threat_profile: str = 'auto',
        random_state: int = 42,
        n_jobs: int = 1,
        if_n_estimators: int = 100,
        lof_n_neighbors: int = 20,
    ):
        self.contamination = contamination
        self.alpha = alpha
        self.voting = voting
        self.threat_profile = threat_profile
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.if_n_estimators = if_n_estimators
        self.lof_n_neighbors = lof_n_neighbors

    # ------------------------------------------------------------------
    # Внутренние вспомогательные методы
    # ------------------------------------------------------------------

    def _resolve_contaminations(self) -> tuple:
        """
        Вычислить эффективные значения contamination для каждого детектора
        в зависимости от threat_profile.

        Возвращает
        ----------
        contamination_if : float
            Contamination для IsolationForest.
        contamination_lof : float
            Contamination для LocalOutlierFactor.
        """
        c = float(self.contamination)
        if self.threat_profile == 'backdoor':
            c_if = min(c * 1.2, 0.499)
            c_lof = min(c * 0.8, 0.499)
        elif self.threat_profile == 'lf':
            c_if = min(c * 0.8, 0.499)
            c_lof = min(c * 1.2, 0.499)
        else:  # 'auto'
            c_if = min(c, 0.499)
            c_lof = min(c, 0.499)
        # Гарантируем минимальный порог
        c_if = max(c_if, 1e-4)
        c_lof = max(c_lof, 1e-4)
        return c_if, c_lof

    @staticmethod
    def _to_numpy(X):
        """Приводит X к numpy array (поддержка DataFrame и sparse-матриц)."""
        # pandas DataFrame
        if hasattr(X, 'values'):
            return X.values
        # scipy sparse
        if hasattr(X, 'toarray'):
            return X.toarray()
        return np.asarray(X)

    def _compute_if_scores(self, X) -> np.ndarray:
        """
        Вычислить нормированные скоры аномальности IsolationForest.
        Высокий скор → аномальный объект.

        Возвращает
        ----------
        scores : np.ndarray, shape (n_samples,)
            Скоры в диапазоне [0, 1].
        """
        raw = self.isolation_forest_.decision_function(X)
        # decision_function: меньше → аномальнее; инвертируем и нормируем
        raw_inv = -raw
        s_min, s_max = raw_inv.min(), raw_inv.max()
        if s_max - s_min < 1e-12:
            return np.zeros(len(raw_inv))
        return (raw_inv - s_min) / (s_max - s_min)

    def _compute_lof_scores(self, X) -> np.ndarray:
        """
        Вычислить нормированные скоры аномальности LocalOutlierFactor.
        Использует negative_outlier_factor_ (хранится после fit).
        Высокий скор → аномальный объект.

        Возвращает
        ----------
        scores : np.ndarray, shape (n_samples,)
            Скоры в диапазоне [0, 1].
        """
        # negative_outlier_factor_: меньше (более отрицательное) → аномальнее
        raw = self.lof_.negative_outlier_factor_
        raw_inv = -raw
        s_min, s_max = raw_inv.min(), raw_inv.max()
        if s_max - s_min < 1e-12:
            return np.zeros(len(raw_inv))
        return (raw_inv - s_min) / (s_max - s_min)

    def _compute_combined_scores(self, X_np: np.ndarray) -> np.ndarray:
        """
        Вычислить составной скор аномальности:
            s(x) = alpha * s_IF + (1 - alpha) * s_LOF

        Параметры
        ----------
        X_np : np.ndarray
            Матрица признаков (уже преобразованная к numpy).

        Возвращает
        ----------
        combined : np.ndarray, shape (n_samples,)
            Составные скоры аномальности в [0, 1].
        """
        s_if = self._compute_if_scores(X_np)
        s_lof = self._compute_lof_scores(X_np)
        return self.alpha * s_if + (1.0 - self.alpha) * s_lof

    def _compute_threshold(self) -> float:
        """
        Вычислить порог отсечения для скора аномальности на основе
        усреднённого contamination.

        Возвращает
        ----------
        threshold : float
            Перцентиль в [0, 1].
        """
        avg_contamination = (self.contamination_if_ + self.contamination_lof_) / 2.0
        # Порог: объекты выше (1 - contamination) перцентиля считаются аномалиями
        return 1.0 - avg_contamination

    def _log_mlflow(self, n_removed: int, n_total: int) -> None:
        """
        Опциональное логирование параметров и метрик в активный MLflow run.
        Вызов безопасен при отсутствии MLflow или активного run.
        """
        try:
            import mlflow
            active_run = mlflow.active_run()
            if active_run is not None:
                mlflow.log_params({
                    'defense_contamination': self.contamination,
                    'defense_alpha': self.alpha,
                    'defense_voting': self.voting,
                    'defense_threat_profile': self.threat_profile,
                    'defense_contamination_if': self.contamination_if_,
                    'defense_contamination_lof': self.contamination_lof_,
                })
                mlflow.log_metrics({
                    'defense_n_removed': n_removed,
                    'defense_removal_rate': n_removed / n_total if n_total > 0 else 0.0,
                    'defense_n_kept': n_total - n_removed,
                })
        except Exception:
            # MLflow недоступен или активного run нет — игнорируем тихо
            pass

    # ------------------------------------------------------------------
    # Основные публичные методы
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """
        Обучить детекторы аномалий на (потенциально отравленной) обучающей выборке.

        Параметры
        ----------
        X : array-like, shape (n_samples, n_features)
            Матрица признаков. Поддерживаются numpy arrays, pandas DataFrames,
            scipy sparse matrices.
        y : игнорируется
            Присутствует для совместимости с sklearn API.

        Возвращает
        ----------
        self : DefenseTransformer
            Обученный трансформер.
        """
        X_np = self._to_numpy(X)
        self.n_samples_fit_ = X_np.shape[0]

        # Уровень 3: вычислить contamination под профиль угрозы
        self.contamination_if_, self.contamination_lof_ = self._resolve_contaminations()

        # Уровень 1: обучить IsolationForest
        self.isolation_forest_ = IsolationForest(
            n_estimators=self.if_n_estimators,
            contamination=self.contamination_if_,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.isolation_forest_.fit(X_np)

        # Уровень 1: обучить LocalOutlierFactor (novelty=False — транзакционный режим)
        self.lof_ = LocalOutlierFactor(
            n_neighbors=self.lof_n_neighbors,
            contamination=self.contamination_lof_,
            novelty=False,
            n_jobs=self.n_jobs,
        )
        self.lof_.fit(X_np)

        return self

    def transform(self, X, y=None):
        """
        Применить фильтрацию: удалить предположительно отравленные объекты.

        Параметры
        ----------
        X : array-like, shape (n_samples, n_features)
            Матрица признаков.
        y : array-like, shape (n_samples,) или None
            Метки классов. Если передан — возвращается очищенная пара (X_clean, y_clean).

        Возвращает
        ----------
        X_clean : np.ndarray
            Очищенная матрица признаков.
        (X_clean, y_clean) : tuple
            Если y is not None — возвращается кортеж очищенных данных.

        Предупреждения
        --------------
        Если после очистки осталось менее 100 объектов — выдаётся предупреждение
        и возвращаются оригинальные данные без фильтрации.
        """
        check_is_fitted(self, ['isolation_forest_', 'lof_'])

        X_np = self._to_numpy(X)
        n_total = X_np.shape[0]

        # LOF в non-novelty режиме использует negative_outlier_factor_ из fit,
        # поэтому для transform нужно переобучить на тех же данных.
        # Для консистентности при transform на тех же данных, что и fit —
        # используем сохранённые атрибуты lof_.
        # При вызове на новых данных — выбрасываем предупреждение.
        if n_total != self.n_samples_fit_:
            warnings.warn(
                f"DefenseTransformer: transform вызван на {n_total} объектах, "
                f"но fit был на {self.n_samples_fit_}. "
                "LOF работает только на обучающих данных (novelty=False). "
                "Рекомендуется использовать fit_transform для полной очистки.",
                UserWarning,
                stacklevel=2,
            )

        removal_mask = self.get_removal_mask(X_np)

        n_removed = int(removal_mask.sum())
        n_kept = n_total - n_removed

        # Краевой случай: слишком мало объектов после очистки
        if n_kept < 100:
            warnings.warn(
                f"DefenseTransformer: после очистки осталось {n_kept} объектов "
                f"(< 100). Фильтрация пропущена, возвращаются оригинальные данные.",
                UserWarning,
                stacklevel=2,
            )
            if y is not None:
                y_np = np.asarray(y)
                return X_np, y_np
            return X_np

        keep_mask = ~removal_mask
        X_clean = X_np[keep_mask]

        # Логирование в MLflow (если доступен и есть активный run)
        self._log_mlflow(n_removed, n_total)

        if y is not None:
            y_np = np.asarray(y)
            y_clean = y_np[keep_mask]
            return X_clean, y_clean

        return X_clean

    def fit_transform(self, X, y=None, **fit_params):
        """
        Обучить детекторы и применить фильтрацию в один шаг.

        Предпочтительный метод для очистки обучающей выборки, так как
        LOF корректно работает только на тех же данных, на которых обучен.

        Параметры
        ----------
        X : array-like, shape (n_samples, n_features)
            Матрица признаков (потенциально отравленная).
        y : array-like, shape (n_samples,) или None
            Метки классов. Если передан — возвращается очищенная пара (X_clean, y_clean).
        **fit_params : dict
            Дополнительные параметры (игнорируются, для совместимости с Pipeline).

        Возвращает
        ----------
        X_clean : np.ndarray
            Очищенная матрица признаков.
        (X_clean, y_clean) : tuple
            Если y is not None.
        """
        return self.fit(X, y).transform(X, y)

    def get_anomaly_scores(self, X) -> np.ndarray:
        """
        Вычислить составные скоры аномальности для каждого объекта.

        Высокий скор означает, что объект с большей вероятностью является
        отравленным. Скоры нормированы в диапазон [0, 1].

        Параметры
        ----------
        X : array-like, shape (n_samples, n_features)
            Матрица признаков.

        Возвращает
        ----------
        scores : np.ndarray, shape (n_samples,)
            Составные скоры аномальности s(x) = alpha * s_IF + (1-alpha) * s_LOF.
        """
        check_is_fitted(self, ['isolation_forest_', 'lof_'])
        X_np = self._to_numpy(X)
        return self._compute_combined_scores(X_np)

    def get_removal_mask(self, X) -> np.ndarray:
        """
        Получить булеву маску объектов, помеченных как отравлённые.

        Параметры
        ----------
        X : array-like, shape (n_samples, n_features)
            Матрица признаков.

        Возвращает
        ----------
        mask : np.ndarray of bool, shape (n_samples,)
            True означает, что объект помечен как отравлённый и должен быть удалён.
        """
        check_is_fitted(self, ['isolation_forest_', 'lof_'])
        X_np = self._to_numpy(X)

        if self.voting == 'hard':
            # Уровень 2, режим 'hard': удалить только если оба детектора флагируют
            if_pred = self.isolation_forest_.predict(X_np)  # -1 = аномалия
            lof_pred = self.lof_.fit_predict(X_np)           # -1 = аномалия
            mask = (if_pred == -1) & (lof_pred == -1)
        else:
            # Уровень 2, режим 'soft': порог по составному скору
            scores = self._compute_combined_scores(X_np)
            threshold = self._compute_threshold()
            # Квантиль-порог: объекты выше threshold-перцентиля
            score_threshold = np.quantile(scores, threshold)
            mask = scores >= score_threshold

        return mask
