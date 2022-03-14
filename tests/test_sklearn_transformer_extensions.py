from io import StringIO
import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal
from pathlib import PosixPath
from sklearn.compose import make_column_transformer as _make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import _get_column_indices
import pandas as pd
import pytest

from sklearn_transformer_extensions import __version__
from sklearn_transformer_extensions import XyAdapter
from sklearn_transformer_extensions import FunctionTransformer
from sklearn_transformer_extensions import make_column_transformer


def test_version():
    assert __version__ == '0.1.9'


@pytest.fixture
def student_records():
    return StringIO("""
Name,OverallGrade,Obedient,ResearchScore,ProjectScore,Recommend
Henry,A,Y,90,85,Yes
John,C,N,85,51,Yes
David,F,N,10,17,No
Holmes,B,Y,75,71,No
Marvin,E,N,20,30,No
Simon,A,Y,92,79,Yes
Robert,B,Y,60,59,No
Trent,C,Y,75,33,No
    """)


def test_xyadapter_return_identical_wotc_for_df(student_records):
    train_o: pd.DataFrame = pd.read_csv(student_records)  # type: ignore
    t1 = XyAdapter(FunctionTransformer(lambda df: df))
    train = t1.fit_transform(train_o)
    assert_frame_equal(train_o, train)


def test_xyadapter_return_identical_wtc_for_df(student_records):
    train_o: pd.DataFrame = pd.read_csv(student_records)  # type: ignore
    t1 = XyAdapter(FunctionTransformer(lambda df: df), target_col='Recommend')
    train = t1.fit_transform(train_o)
    assert_frame_equal(train_o, train)


def test_xyadapter_return_identical_wotc_for_numpy(student_records):
    train_o: pd.DataFrame = pd.read_csv(student_records)  # type: ignore
    t1 = XyAdapter(FunctionTransformer(lambda X: X))
    train = t1.fit_transform(train_o.to_numpy())
    assert_array_equal(train_o.to_numpy(), train)


def test_xyadapter_return_identical_wtc_for_numpy(student_records):
    train_o: pd.DataFrame = pd.read_csv(student_records)  # type: ignore
    t1 = XyAdapter(FunctionTransformer(lambda X: X), target_col=5)
    train = t1.fit_transform(train_o.to_numpy())
    assert_array_equal(train_o.to_numpy(), train)


def test_xyadapter_columntrans_pipeline_fit_for_df(student_records):
    train_o: pd.DataFrame = pd.read_csv(student_records)  # type: ignore
    ct1 = XyAdapter(
        _make_column_transformer(
            (StandardScaler(), ["ResearchScore", "ProjectScore"]),
            (OneHotEncoder(handle_unknown='ignore'),
             ['OverallGrade', 'Obedient']),
            verbose_feature_names_out=False,
        ), target_col='Recommend')
    train = ct1.fit_transform(train_o)

    engineered_features = [
        'ResearchScore', 'ProjectScore', 'OverallGrade_A', 'OverallGrade_B',
        'OverallGrade_C', 'OverallGrade_E', 'OverallGrade_F', 'Obedient_N',
        'Obedient_Y'
    ]
    expected_columns = engineered_features + ['Recommend']

    assert isinstance(train, pd.DataFrame)
    assert train.shape[0] == train_o.shape[0]
    assert train.shape[1] == len(expected_columns)
    assert train.columns.tolist() == expected_columns
    assert all(train[engineered_features].dtypes == float)
    assert_array_almost_equal(train[engineered_features].sum(axis=0),
                              np.r_[0, 0, 2, 2, 2, 1, 1, 3, 5])


def test_xyadapter_columntrans_pipeline_fit_for_numpy(student_records):
    train_o: pd.DataFrame = pd.read_csv(student_records)  # type: ignore
    ct1 = XyAdapter(
        _make_column_transformer(
            (StandardScaler(),
             _get_column_indices(train_o, ["ResearchScore", "ProjectScore"])),
            (OneHotEncoder(handle_unknown='ignore'),
             _get_column_indices(train_o, ["OverallGrade", "Obedient"])),
            verbose_feature_names_out=False,
        ), target_col=5)
    train = ct1.fit_transform(train_o.to_numpy())

    engineered_features = [
        'ResearchScore', 'ProjectScore', 'OverallGrade_A', 'OverallGrade_B',
        'OverallGrade_C', 'OverallGrade_E', 'OverallGrade_F', 'Obedient_N',
        'Obedient_Y'
    ]

    expected_columns = len(engineered_features) + 1

    assert isinstance(train, np.ndarray)
    assert train.shape[0] == train_o.shape[0]
    assert train.shape[1] == expected_columns
    assert train[:, :-1].dtype == np.dtype('O')
    assert_array_almost_equal(train[:, :-1].sum(axis=0), np.r_[0, 0, 2, 2, 2, 1,
                                                               1, 3, 5])


def test_model_build_eval_deploy(student_records, tmpdir):
    train_o: pd.DataFrame = pd.read_csv(student_records)  # type: ignore
    ct1 = XyAdapter(
        _make_column_transformer(
            (StandardScaler(),
             _get_column_indices(train_o, ["ResearchScore", "ProjectScore"])),
            (OneHotEncoder(handle_unknown='ignore'),
             _get_column_indices(train_o, ["OverallGrade", "Obedient"])),
            verbose_feature_names_out=False,
        ), target_col='Recommend')

    lr = XyAdapter(LogisticRegression(), target_col='Recommend')

    p = make_pipeline(ct1, lr).fit(train_o)
    pred_labels = p.predict(train_o)
    actual_labels = train_o['Recommend']

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report

    assert accuracy_score(actual_labels, pred_labels) == 1.0
    assert classification_report(actual_labels, pred_labels) == """\
              precision    recall  f1-score   support

          No       1.00      1.00      1.00         5
         Yes       1.00      1.00      1.00         3

    accuracy                           1.00         8
   macro avg       1.00      1.00      1.00         8
weighted avg       1.00      1.00      1.00         8
"""

    # Deploy
    import joblib
    joblib.dump(p, str(tmpdir / 'pipeline.pkl'))

    # Serve
    p = joblib.load(str(tmpdir / 'pipeline.pkl'))

    new_data = pd.DataFrame([{
        'Name': 'Nathan',
        'OverallGrade': 'F',
        'Obedient': 'N',
        'ResearchScore': 30,
        'ProjectScore': 20
    }, {
        'Name': 'Thomas',
        'OverallGrade': 'A',
        'Obedient': 'Y',
        'ResearchScore': 78,
        'ProjectScore': 80
    }])
    new_data = new_data[[
        'Name', 'OverallGrade', 'Obedient', 'ResearchScore', 'ProjectScore'
    ]]

    assert_frame_equal(
        p[:-1].transform(new_data),
        pd.read_csv(
            StringIO("""
ResearchScore,ProjectScore,OverallGrade_A,OverallGrade_F,Obedient_N,Obedient_Y,OverallGrade_B,OverallGrade_E,OverallGrade_C
-1.127647,-1.430636,0.,1.,1.,0.,0.,0.,0.
0.494137,1.160705,1.,0.,0.,1.,0.,0.,0.
        """)), check_like=True)

    predictions = p.predict(new_data)
    assert predictions.tolist() == ['No', 'Yes']


def test_reg_ft_reg_ct(student_records):

    train_o: pd.DataFrame = pd.read_csv(student_records)  # type: ignore
    f1 = FunctionTransformer(lambda df: df)
    assert_frame_equal(train_o, f1.fit_transform(train_o))
    assert_frame_equal(train_o, f1.transform(train_o))

    ct = _make_column_transformer(
        (FunctionTransformer(lambda df: df), lambda df: df.columns.tolist()),
        remainder='drop',
        verbose_feature_names_out=False,
    )
    ret = ct.fit_transform(train_o)
    assert_array_equal(train_o.to_numpy(), ret)  # type: ignore

    ct = _make_column_transformer(
        (FunctionTransformer(lambda df: df), ['ResearchScore']),
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    ret = ct.fit_transform(train_o)
    exp = train_o.iloc[:, [3, 0, 1, 2, 4, 5]].to_numpy()
    assert_array_equal(exp, ret)  # type: ignore


@pytest.mark.parametrize("make_ct",
                         [make_column_transformer, _make_column_transformer])
def test_ext_ft_reg_ct_fitfn(student_records, make_ct):

    train_o: pd.DataFrame = pd.read_csv(student_records)  # type: ignore

    fit_fn = lambda *_: dict(p=1)
    ct = make_ct(
        (FunctionTransformer(lambda df, p: df + p,
                             fit_fn=fit_fn), ['ResearchScore']),
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    ret = ct.fit_transform(train_o)
    exp = train_o.iloc[:, [3, 0, 1, 2, 4, 5]].to_numpy()
    exp[:, 0] += 1
    assert_array_equal(exp, ret)  # type: ignore

    fit_fn = lambda *_: 1
    ct = make_ct(
        (FunctionTransformer(lambda df, params: df + params,
                             fit_fn=fit_fn), ['ResearchScore']),
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    ret = ct.fit_transform(train_o)
    exp = train_o.iloc[:, [3, 0, 1, 2, 4, 5]].to_numpy()
    exp[:, 0] += 1
    assert_array_equal(exp, ret)  # type: ignore


@pytest.mark.parametrize("make_ct",
                         [make_column_transformer, _make_column_transformer])
def test_ext_ft_reg_ct_colname(student_records, make_ct):

    train_o: pd.DataFrame = pd.read_csv(student_records)  # type: ignore

    ct = make_ct(
        (FunctionTransformer(lambda df: df), ['ResearchScore']),
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    ct.fit(train_o)
    ret = ct.get_feature_names_out().tolist()
    exp = [
        'ResearchScore', 'Name', 'OverallGrade', 'Obedient', 'ProjectScore',
        'Recommend'
    ]
    assert ret == exp

    col_fn = lambda cols: [col + '_E' for col in cols]
    ct = make_ct(
        (FunctionTransformer(lambda df: df, col_name=col_fn), ['ResearchScore'
                                                               ]),
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    ct.fit(train_o)
    ret = ct.get_feature_names_out().tolist()
    if make_ct == _make_column_transformer:
        exp = [
            'ResearchScore_E', 'Name', 'OverallGrade', 'Obedient',
            'ProjectScore', 'Recommend'
        ]
    else:
        exp = [
            'ResearchScore_E', 'Name', 'OverallGrade', 'Obedient',
            'ResearchScore', 'ProjectScore', 'Recommend'
        ]
    assert ret == exp

    col_fn = 'new_col'
    ct = make_ct(
        (FunctionTransformer(lambda df: df, col_name=col_fn), ['ResearchScore'
                                                               ]),
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    ct.fit(train_o)
    ret = ct.get_feature_names_out().tolist()
    if make_ct == _make_column_transformer:
        exp = [
            'new_col', 'Name', 'OverallGrade', 'Obedient', 'ProjectScore',
            'Recommend'
        ]
    else:
        exp = [
            'new_col', 'Name', 'OverallGrade', 'Obedient', 'ResearchScore',
            'ProjectScore', 'Recommend'
        ]
    assert ret == exp

    col_fn = lambda cols: [col + '_E' for col in cols]
    ct = make_ct(
        (FunctionTransformer(
            lambda df: df, col_name=col_fn), ['ResearchScore', 'ProjectScore']),
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    ct.fit(train_o)
    ret = ct.get_feature_names_out().tolist()
    if make_ct == _make_column_transformer:
        exp = [
            'ResearchScore_E', 'ProjectScore_E', 'Name', 'OverallGrade',
            'Obedient', 'Recommend'
        ]
    else:
        exp = [
            'ResearchScore_E', 'ProjectScore_E', 'Name', 'OverallGrade',
            'Obedient', 'ResearchScore', 'ProjectScore', 'Recommend'
        ]
    assert ret == exp

    col_fn = ['new_col1', 'new_col2']
    ct = make_ct(
        (FunctionTransformer(
            lambda df: df, col_name=col_fn), ['ResearchScore', 'ProjectScore']),
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    ct.fit(train_o)
    ret = ct.get_feature_names_out().tolist()
    if make_ct == _make_column_transformer:
        exp = [
            'new_col1', 'new_col2', 'Name', 'OverallGrade', 'Obedient',
            'Recommend'
        ]
    else:
        exp = [
            'new_col1', 'new_col2', 'Name', 'OverallGrade', 'Obedient',
            'ResearchScore', 'ProjectScore', 'Recommend'
        ]
    assert ret == exp
