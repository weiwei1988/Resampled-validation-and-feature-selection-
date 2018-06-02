<h1> リサンプリングを考慮した交差検証/特徴量選択</h1>

<h2>背景</h2>
<p>製造機械の故障検知、クレジットカードの不正利用検知など、機械学習における「異常検知」の分野では、正列と負列のサンプル数が不均衡なデータを扱うことが多い。これら不均衡データを取り扱う場合しばしば実施されるのがデータのリサンプリングである。Pythonライブラリーでは、<a herf="https://github.com/scikit-learn-contrib/imbalanced-learn.git">imbalanced-learnライブラリー</a>内で利用可能なU<a herf="http://contrib.scikit-learn.org/imbalanced-learn/stable/api.html#module-imblearn.under_sampling">Undersampling</a>, <a herf="http://contrib.scikit-learn.org/imbalanced-learn/stable/api.html#module-imblearn.over_sampling">OverSampling</a>などの手法を使ったクラスがあるが、これらの関数はTrain Data１つずつしか適用できない。また<a herf="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html">scikit-learn.model_selectionで利用可能なcross_validateクラス</a>は、初期化時点で代入したデータを分割して交差検証を実施してしまうため、これらを利用するだけでは「Train DataとTest Dataの分割⇒Train Dataをリサンプリング⇒モデルで学習⇒Test Dataで精度検証⇒次のTrain, Testセットについて同様のことを実施」という、データのリサンプリングプロセスを組み込んだ交差検証が実施できない。</p>

<p>同様にして、<a herf="http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html">scikit-learn.feature_selection内にある特徴量逐次削減クラスであるRFECV</a>も同じ問題を抱えており、データリサンプリング作業を組み込んだ特徴量選択が実施できない</p>

<p>上記の問題を解決するため、データのリサンプリングプロセスを交差検証に組み込んだクラスと、データリサンプリングプロセスを特徴量逐次削減に組み込んだクラスを実装し、インポート可能なモジュールとして容易した</p>

<h2>本リポジトリで用意したモジュール</h2>
<ol>
<li> Resampled_learn.py (二値分類用モジュール)</li>
<li> Resampled_learn_multiclass.py (他クラス用モジュール)</li>
</ol>

<h2>使い方</h2>
<p>後日追加</p>