# coding: utf-8

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, log_loss
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

"""警告の非表示"""
import warnings
warnings.filterwarnings('ignore')


class Resampled_Prediction:

    def __init__(self,
                 sampler=RandomUnderSampler(ratio='not minority'),
                 scaler=StandardScaler(),
                 estimator=xgb.XGBClassifier(),
                 verbose=True
                 ):

        self.sampler = sampler
        self.scaler = scaler
        self.estimator = estimator
        self.verbose = verbose
        self.feature_importances_ = 'No Value'

    def fit(self, X, y):
        if self.sampler is not None:
            X_resampled, y_resampled = self.sampler.fit_sample(X, y)
        else:
            X_resampled, y_resampled = X, y

        if self.scaler is not None:
            self.scaler.fit(X_resampled)
            X_resampled = self.scaler.transform(X_resampled)
        else:
            pass

        self.estimator.fit(X_resampled, y_resampled)

        if hasattr(self.estimator, 'feature_importances_') is True:
            self.feature_importances_ = self.estimator.feature_importances_

        elif hasattr(self.estimator, 'coef_') is True:
            self.coef_ = self.estimator.coef_

    def predict(self, X_test):

        if self.scaler is not None:
            return self.estimator.predict(self.scaler.transform(X_test))
        else:
            return self.estimator.predict(X_test)

    def predict_proba(self, X_test):

        if self.scaler is not None:
            return self.estimator.predict_proba(self.scaler.transform(X_test))
        else:
            return self.estimator.predict_proba(X_test)

    def score(self, X, y):

        if self.scaler is not None:
            return self.estimator.score(self.scaler.transform(X), y)
        else:
            return self.estimator.score(X, y)


class Resampled_Cross_Validate:

    def __init__(self,
                 cv,
                 sampler=RandomOverSampler(ratio='not minority'),
                 scaler=StandardScaler(),
                 estimator=xgb.XGBClassifier(),
                 verbose=True
                 ):

        self.cv = cv
        self.verbose = verbose
        self.sampler = sampler
        self.scaler = scaler
        self.estimator = estimator

        self.Matrix = 'No Value'
        self.acc_ = 'No Value'
        self.pre_ = 'No Value'
        self.rec_ = 'No Value'
        self.f1_ = 'No Value'
        self.roc_auc_ = 'No Value'
        self.logloss = 'No Value'

    def fit(self, X_train, y_train):

        matrix = []
        ACC = []
        PRE = []
        REC = []
        F1 = []
        ROC_AUC = []
        logloss = []
        k = 1

        flod = StratifiedKFold(n_splits=self.cv)

        if self.verbose is True:
            print("Start Processing Resampled Validation: %d splits" % self.cv)
        else:
            pass

        for train_index, test_index in flod.split(X_train, y_train):
            x_ta = X_train.values[train_index]
            x_te = X_train.values[test_index]
            y_ta = y_train.values[train_index]
            y_te = y_train.values[test_index]

            try:
                if self.sampler is not None:
                    x_ta_resampled, y_ta_resampled = self.sampler.fit_sample(
                        x_ta, y_ta)
                else:
                    x_ta_resampled, y_ta_resampled = x_ta, y_ta

            except ValueError:
                print(
                    'Error on Sampler. Please use imblearn-RandomUndersampler, RandomOverSampler or Combined methods')

            if self.scaler is not None:
                pipe = make_pipeline(self.scaler, self.estimator)
            else:
                pipe = self.estimator

            try:
                pipe.fit(x_ta_resampled, y_ta_resampled)
                y_pred = pipe.predict(x_te)
                y_prob = pipe.predict_proba(x_te)

            except ValueError:
                print(
                    'Error on estimator, Please use right estimator for binary classification')

            matrix.append(confusion_matrix(y_te, y_pred))
            ACC.append(accuracy_score(y_te, y_pred))
            PRE.append(precision_score(y_te, y_pred))
            REC.append(recall_score(y_te, y_pred))
            F1.append(f1_score(y_te, y_pred))
            ROC_AUC.append(roc_auc_score(y_te, y_pred))
            logloss.append(log_loss(y_te, y_prob))

            if self.verbose is True:
                print("Done: %d, Totaling: %d" % (k, self.cv))
            else:
                pass

            k += 1

        self.Matrix = matrix
        self.acc_ = np.array(ACC)
        self.pre_ = np.array(PRE)
        self.rec_ = np.array(REC)
        self.f1_ = np.array(F1)
        self.roc_auc_ = np.array(ROC_AUC)
        self.logloss_ = np.array(logloss)


def get_importance_score(X, IM_score):

    Questions = pd.DataFrame(X.columns)
    importance = pd.DataFrame(IM_score)
    Score = pd.concat([Questions, importance], axis=1)
    Score.columns = ['Var', 'Score']
    return Score


def Resampled_Valudation_Score(X_train, y_train, n_splits, sampler, scaler, estimator, verbose=False):

    ACC = []
    PRE = []
    REC = []
    F1 = []
    ROC_AUC = []
    Importance_Score = []
    logloss = []
    k = 1

    flod = StratifiedKFold(n_splits=n_splits)

    for train_index, test_index in flod.split(X_train, y_train):
        x_ta = X_train.values[train_index]
        x_te = X_train.values[test_index]
        y_ta = y_train.values[train_index]
        y_te = y_train.values[test_index]

        try:
            if sampler is not None:
                x_ta_resampled, y_ta_resampled = sampler.fit_sample(x_ta, y_ta)
            else:
                x_ta_resampled, y_ta_resampled = x_ta, y_ta

        except ValueError:
            print(
                'Error on Sampler. Please use imblearn-RandomUndersampler, RandomOverSampler or Combined methods')

        if scaler is not None:
            scaler.fit(x_ta_resampled)
            x_ta_resampled = scaler.transform(x_ta_resampled)
            x_te = scaler.transform(x_te)
        else:
            pass

        try:
            estimator.fit(x_ta_resampled, y_ta_resampled)
            y_pred = estimator.predict(x_te)
            y_prob = estimator.predict_proba(x_te)

        except ValueError:
            print(
                'Error on estimator. Please use right estimator for binary classification')

        ACC.append(accuracy_score(y_te, y_pred))
        ROC_AUC.append(roc_auc_score(y_te, y_pred))
        F1.append(f1_score(y_te, y_pred))
        PRE.append(precision_score(y_te, y_pred))
        REC.append(recall_score(y_te, y_pred))
        logloss.append(log_loss(y_te, y_prob))

        try:
            if hasattr(estimator, 'feature_importances_') is True:
                Importance_Score.append(estimator.feature_importances_)

            elif hasattr(estimator, 'coef_') is True:
                feature_im = np.abs(estimator.coef_).ravel()
                Importance_Score.append(feature_im)

            elif hasattr(estimator, 'dual_coef_') is True:
                w = np.matmul(estimator.dual_coef_,
                              estimator.support_vectors_).transpose()
                feature_im = np.abs(w)
                Importance_Score.append(feature_im)

        except ValueError:
            print('Error on getting feature importance. Please use estimators with atrribute "coef_" or "feature_importances_"')

        if verbose is True:
            print("Done: %d, Totaling: %d" % (k, n_splits))
        else:
            pass

        k += 1

        out_score = get_importance_score(
            X_train, sum(Importance_Score) / n_splits)

    return np.array(ACC), np.array(ROC_AUC), np.array(F1), np.array(PRE), np.array(REC), np.array(logloss), out_score


class Resampled_RFECV:

    def __init__(self,
                 n_steps,
                 cv,
                 sampler=RandomOverSampler(ratio='not minority'),
                 scaler=StandardScaler(),
                 estimator=xgb.XGBClassifier(),
                 verbose=False
                 ):

        self.n_steps = n_steps
        self.cv = cv
        self.sampler = sampler
        self.scaler = scaler
        self.estimator = estimator
        self.verbose = verbose

        self.mean_score_ = 'No Value'
        self.std_score_ = 'No Value'
        self.questions_ = 'No Value'

    def fit(self, X, y):

        if len(X.columns) % self.n_steps != 0:
            print("Error: n_steps must be a divisior of %d" % len(X.columns))
            return 'Error'
        else:
            "結果格納用リストの生成"
            ACC_SCORE_mean = []
            ROC_AUC_mean = []
            F1_SCORE_mean = []
            PRE_SCORE_mean = []
            REC_SCORE_mean = []
            logloss_mean = []

            ACC_SCORE_std = []
            ROC_AUC_std = []
            F1_SCORE_std = []
            PRE_SCORE_std = []
            REC_SCORE_std = []
            logloss_std = []

            Questions = []

            "説明変数の初期化"
            X_new = X

            "計算ステップリストの用意"
            step = np.arange(self.n_steps, len(X.columns) +
                             self.n_steps, self.n_steps)[::-1]

            if self.verbose is True:
                print(
                    "Start Processing Resampled Feature Selection: %d Steps" % len(step))
            else:
                pass

            for i in tqdm(range(len(step))):

                if self.verbose is True:
                    print("Fitting: %d features" % step[i])
                else:
                    pass

                ACC, ROC_AUC, F1, PRE, REC, logloss, IM_score = Resampled_Valudation_Score(X_new, y,
                                                                                           sampler=self.sampler,
                                                                                           scaler=self.scaler,
                                                                                           estimator=self.estimator,
                                                                                           n_splits=self.cv,
                                                                                           verbose=self.verbose)
                IM_score = IM_score.sort_values(
                    by='Score').reset_index(drop=True)
                IM_new = IM_score.drop(range(self.n_steps))

                ACC_SCORE_mean.append(ACC.mean())
                ROC_AUC_mean.append(ROC_AUC.mean())
                F1_SCORE_mean.append(F1.mean())
                PRE_SCORE_mean.append(PRE.mean())
                REC_SCORE_mean.append(REC.mean())
                logloss_mean.append(logloss.mean())

                ACC_SCORE_std.append(ACC.std())
                ROC_AUC_std.append(ROC_AUC.std())
                F1_SCORE_std.append(F1.std())
                PRE_SCORE_std.append(PRE.std())
                REC_SCORE_std.append(REC.std())
                logloss_std.append(logloss.std())

                X_new = X_new.loc[:, IM_new.Var]

                Questions.append(IM_score)

            self.mean_score_ = {
                'ACC': np.array(ACC_SCORE_mean[::-1]),
                'ROC_AUC': np.array(ROC_AUC_mean[::-1]),
                'F1': np.array(F1_SCORE_mean[::-1]),
                'PRE': np.array(PRE_SCORE_mean[::-1]),
                'REC': np.array(REC_SCORE_mean[::-1]),
                'logloss': np.array(logloss_mean[::-1])
            }

            self.std_score_ = {
                'ACC': np.array(ACC_SCORE_std[::-1]),
                'ROC_AUC': np.array(ROC_AUC_std[::-1]),
                'F1': np.array(F1_SCORE_std[::-1]),
                'PRE': np.array(PRE_SCORE_std[::-1]),
                'REC': np.array(REC_SCORE_std[::-1]),
                'logloss': np.array(logloss_std[::-1])
            }

            self.questions_ = Questions[::-1]

    def select_num_Q(self, threshold, score='ROC_AUC'):
        try:
            if score != 'logloss':
                Num_Q = np.where(self.mean_score_[score] > threshold)[0][0] + 1
                return Num_Q
            else:
                Num_Q = np.where(self.mean_score_[score] < threshold)[0][0] + 1
                return Num_Q
        except ValueError:
            print('Error')

    def draw_figure(self, X, y, ymin=0.0, ymax=1.0, fill_btw=True):
        """設問数と精度の関係を描画"""
        plt.clf()
        fig = plt.figure(figsize=(8, 5), facecolor='w')

        plt.plot(np.arange(self.n_steps, len(X.columns) + self.n_steps,
                           self.n_steps), self.mean_score_['ACC'], '-', label='Accuracy')
        plt.plot(np.arange(self.n_steps, len(X.columns) + self.n_steps,
                           self.n_steps), self.mean_score_['ROC_AUC'], '-', label='ROC AUC')
        plt.plot(np.arange(self.n_steps, len(X.columns) + self.n_steps,
                           self.n_steps), self.mean_score_['F1'], '--', label='F1 Score')
        plt.plot(np.arange(self.n_steps, len(X.columns) + self.n_steps,
                           self.n_steps), self.mean_score_['PRE'], '--', label='Precision Score')
        plt.plot(np.arange(self.n_steps, len(X.columns) + self.n_steps,
                           self.n_steps), self.mean_score_['REC'], '--', label='Recall Score')
        plt.plot(np.arange(self.n_steps, len(X.columns) + self.n_steps,
                           self.n_steps), self.mean_score_['logloss'], '--', label='logloss')

        if fill_btw is True:
            plt.fill_between(np.arange(self.n_steps, len(X.columns) + self.n_steps, self.n_steps),
                             self.mean_score_['ACC'] + self.std_score_['ACC'],
                             self.mean_score_['ACC'] - self.std_score_['ACC'],
                             alpha=0.15)

            plt.fill_between(np.arange(self.n_steps, len(X.columns) + self.n_steps, self.n_steps),
                             self.mean_score_['ROC_AUC'] +
                             self.std_score_['ROC_AUC'],
                             self.mean_score_['ROC_AUC'] -
                             self.std_score_['ROC_AUC'],
                             alpha=0.15)

            plt.fill_between(np.arange(self.n_steps, len(X.columns) + self.n_steps, self.n_steps),
                             self.mean_score_['F1'] + self.std_score_['F1'],
                             self.mean_score_['F1'] - self.std_score_['F1'],
                             alpha=0.15)

            plt.fill_between(np.arange(self.n_steps, len(X.columns) + self.n_steps, self.n_steps),
                             self.mean_score_['PRE'] + self.std_score_['PRE'],
                             self.mean_score_['PRE'] - self.std_score_['PRE'],
                             alpha=0.15)

            plt.fill_between(np.arange(self.n_steps, len(X.columns) + self.n_steps, self.n_steps),
                             self.mean_score_['REC'] + self.std_score_['REC'],
                             self.mean_score_['REC'] - self.std_score_['REC'],
                             alpha=0.15)

            plt.fill_between(np.arange(self.n_steps, len(X.columns) + self.n_steps, self.n_steps),
                             self.mean_score_['logloss'] +
                             self.std_score_['logloss'],
                             self.mean_score_['logloss'] -
                             self.std_score_['logloss'],
                             alpha=0.15)
        else:
            pass

        plt.xlabel('No. of Features Selected', fontsize=12)
        plt.ylabel('Validation Score (CV=%d)' % self.cv, fontsize=12)
        plt.title('Score curve', fontsize=14)
        plt.ylim(ymin, ymax)
        plt.legend(loc='best', fontsize=8)
        plt.show()

    def draw_barchart(self, X, y):

        df = self.questions_[len(X.columns) - 1]
        df = df.sort_values(by='Score', ascending=True)

        plt.clf()
        fig = plt.figure(figsize=(8, 5), facecolor='w')

        plt.barh(range(len(X.columns)), df.Score, align='center', color='r')
        plt.xticks(fontsize=10)
        plt.yticks(range(len(X.columns)), df.Var, fontsize=8)

        plt.xlabel('Feature Importance Score', fontsize=12)
        plt.ylabel('Questions', fontsize=12)
        plt.title('Feature imporance Chart')

        plt.show()


class BalancedBagging_Valudation:

    def __init__(self, cv, verbose=True, n_jobs=1, n_estimators=10):
        self.cv = cv
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators

        self.Matrix = 'No Value'
        self.acc_ = 'No Value'
        self.pre_ = 'No Value'
        self.rec_ = 'No Value'
        self.f1_ = 'No Value'
        self.roc_auc_ = 'No Value'
        self.logloss_ = 'No Value'

        self.X_set = 'No Value'
        self.Y_set = 'No Value'

        self.flod = StratifiedKFold(self.cv)

    def fit(self, X_train, y_train):

        matrix = []
        ACC = []
        PRE = []
        REC = []
        F1 = []
        ROC_AUC = []
        logloss = []

        test_set_X = []
        test_set_Y = []

        k = 1

        if self.verbose is True:
            print(
                "Checking Cross Validation Score with Balanced Bagging: %d splits" % self.cv)
        else:
            pass

        for train_index, test_index in self.flod.split(X_train, y_train):
            x_ta = X_train.values[train_index]
            x_te = X_train.values[test_index]
            y_ta = y_train.values[train_index]
            y_te = y_train.values[test_index]

            sts = StandardScaler()
            clf = xgb.XGBClassifier(n_jobs=self.n_jobs)
            usbc = BalancedBaggingClassifier(
                base_estimator=clf, n_jobs=self.n_jobs, n_estimators=self.n_estimators, ratio='not minority')
            pipe = make_pipeline(sts, usbc)

            pipe.fit(x_ta, y_ta)
            y_pred = pipe.predict(x_te)
            y_prob = pipe.predict_proba(x_te)

            matrix.append(confusion_matrix(y_te, y_pred))
            ACC.append(accuracy_score(y_te, y_pred))
            PRE.append(precision_score(y_te, y_pred))
            REC.append(recall_score(y_te, y_pred))
            F1.append(f1_score(y_te, y_pred))
            ROC_AUC.append(roc_auc_score(y_te, y_pred))
            logloss.append(log_loss(y_te, y_prob))

            test_set_X.append(x_ta)
            test_set_Y.append(y_ta)

            if self.verbose is True:
                print("Done: %d, Totaling: %d" % (k, self.cv))
            else:
                pass

            k += 1

        self.Matrix = matrix
        self.acc_ = np.array(ACC)
        self.pre_ = np.array(PRE)
        self.rec_ = np.array(REC)
        self.f1_ = np.array(F1)
        self.roc_auc_ = np.array(ROC_AUC)
        self.logloss_ = np.array(logloss)

        self.X_set = test_set_X
        self.Y_set = test_set_Y

    def predict(self, X_test):
        best_estimator = np.where(self.acc_ == self.acc_.max())[0][0]

        sts = StandardScaler()
        clf = xgb.XGBClassifier(n_jobs=self.n_jobs)
        usbc = BalancedBaggingClassifier(
            base_estimator=clf, n_jobs=self.n_jobs, n_estimators=self.n_estimators, ratio='not minority')
        pipe = make_pipeline(sts, usbc)

        pipe.fit(self.X_set[best_estimator], self.Y_set[best_estimator])
        Y_pred = pipe.predict(X_test)

        return Y_pred


def Check_TestData(X_train, y_train, X_test, y_test):
    matrix = []
    PRE = []
    REC = []
    F1 = []
    ROC_AUC = []
    logloss = []

    print("Checking Test Score with Balanced Bagging")

    sts = StandardScaler()
    clf = xgb.XGBClassifier(n_jobs=-1)
    usbc = BalancedBaggingClassifier(
        base_estimator=clf, n_jobs=-1, n_estimators=10, ratio='not minority')
    pipe = make_pipeline(sts, usbc)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)

    matrix = confusion_matrix(y_test, y_pred)
    PRE = precision_score(y_test, y_pred)
    REC = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    ROC_AUC = roc_auc_score(y_test, y_pred)
    logloss = np.array(log_loss(y_test, y_prob))

    return matrix, PRE, REC, F1, ROC_AUC, logloss


class Resampled_RFE:

    def __init__(self,
                 n_feature_select,
                 n_steps,
                 cv,
                 sampler=RandomOverSampler(ratio='not minority'),
                 scaler=StandardScaler(),
                 estimator=xgb.XGBClassifier(),
                 verbose=False
                 ):

        self.n_steps = n_steps
        self.n_feature_select = n_feature_select
        self.cv = cv
        self.sampler = sampler
        self.scaler = scaler
        self.estimator = estimator
        self.verbose = verbose

        self.N_feature = 'No Value'
        self.n_feature_reduce = 'No Value'

        self.mean_score_ = 'No Value'
        self.std_score_ = 'No Value'
        self.questions_ = 'No Value'

    def fit(self, X, y):
        self.N_feature = len(X.columns)
        self.n_feature_reduce = self.N_feature - self.n_feature_select

        if self.n_feature_reduce % self.n_steps != 0:
            raise ValueError(
                'Error: n_steps must be a divisior of %d' % self.n_feature_reduce)
        else:
            "結果格納用リストの生成"
            ACC_SCORE_mean = []
            ROC_AUC_mean = []
            F1_SCORE_mean = []
            PRE_SCORE_mean = []
            REC_SCORE_mean = []
            logloss_mean = []

            ACC_SCORE_std = []
            ROC_AUC_std = []
            F1_SCORE_std = []
            PRE_SCORE_std = []
            REC_SCORE_std = []
            logloss_std = []

            Questions = []

            "説明変数の初期化"
            X_new = X

            "計算ステップリストの用意"
            step = np.arange(self.n_steps, self.n_feature_reduce +
                             self.n_steps, self.n_steps)[::-1]

            if self.verbose is True:
                print(
                    "Start Processing Resampled Feature Selection: %d Steps" % len(step))
            else:
                pass

            for i in tqdm(range(len(step))):

                if self.verbose is True:
                    print("Fitting: %d features" % step[i])
                else:
                    pass

                ACC, ROC_AUC, F1, PRE, REC, logloss, IM_score = Resampled_Valudation_Score(X_new, y,
                                                                                           sampler=self.sampler,
                                                                                           scaler=self.scaler,
                                                                                           estimator=self.estimator,
                                                                                           n_splits=self.cv,
                                                                                           verbose=self.verbose)
                if i == 0:
                    Questions.append(IM_score)

                IM_score = IM_score.sort_values(
                    by='Score').reset_index(drop=True)
                IM_new = IM_score.drop(range(self.n_steps))

                ACC_SCORE_mean.append(ACC.mean())
                ROC_AUC_mean.append(ROC_AUC.mean())
                F1_SCORE_mean.append(F1.mean())
                PRE_SCORE_mean.append(PRE.mean())
                REC_SCORE_mean.append(REC.mean())
                logloss_mean.append(logloss.mean())

                ACC_SCORE_std.append(ACC.std())
                ROC_AUC_std.append(ROC_AUC.std())
                F1_SCORE_std.append(F1.std())
                PRE_SCORE_std.append(PRE.std())
                REC_SCORE_std.append(REC.std())
                logloss_std.append(logloss.std())

                X_new = X_new.loc[:, IM_new.Var]
                Questions.append(IM_new)

            self.mean_score_ = {
                'ACC': np.array(ACC_SCORE_mean[::-1]),
                'ROC_AUC': np.array(ROC_AUC_mean[::-1]),
                'F1': np.array(F1_SCORE_mean[::-1]),
                'PRE': np.array(PRE_SCORE_mean[::-1]),
                'REC': np.array(REC_SCORE_mean[::-1]),
                'logloss': np.array(logloss_mean[::-1])
            }

            self.std_score_ = {
                'ACC': np.array(ACC_SCORE_std[::-1]),
                'ROC_AUC': np.array(ROC_AUC_std[::-1]),
                'F1': np.array(F1_SCORE_std[::-1]),
                'PRE': np.array(PRE_SCORE_std[::-1]),
                'REC': np.array(REC_SCORE_std[::-1]),
                'logloss': np.array(logloss_std[::-1])
            }

            self.questions_ = Questions[::-1]

    def support(self):
        df_t = pd.merge(self.questions_[self.n_feature_reduce], self.questions_[
                        0], on='Var', how='outer')
        df_result = pd.concat([df_t['Var'], df_t['Score_y'].notnull()], axis=1)
        df_result.columns = ['Var', 'Support']

        return df_result
