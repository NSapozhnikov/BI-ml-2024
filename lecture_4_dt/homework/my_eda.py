"""
automatisation of EDA
"""
import random
import pandas as pd


def stat_categorical(column: pd.Series) -> dict:
    """
    calculate statistics for categorical columns

    Args:
    - column - a column to check

    Returns:
    dict with statistics
    """
    unique_values = column.nunique()
    total_values = column.count()
    ratio = unique_values / total_values

    if ratio < 0.05 and unique_values < 100:
        return {'unique_values': unique_values,
                'total_values': total_values,
                'ratio': ratio}


def stat_noncategorical(column: pd.Series) -> dict:
    """
    calculate statistics for non-categorical columns
    min, max, mean, std, q0.25, median, q.075

    Args:
    - column - a column to check

    Returns:
    dict with statistics
    """
    return {'min': column.min(),
            'max': column.max(),
            'mean': column.mean(),
            'std': column.std(),
            'q25': column.quantile(q=0.25),
            'median': column.median(),
            'q75': column.quantile(q=0.75)}


def run_eda(df: pd.DataFrame) -> None:
    """
    main EDA function
    """
    print('pRiVeT)',
          ''.join(random.choice(')0') for _ in range(9)), '!', sep='')
    print('Number of observations (rows) is:', df.shape[0], sep='\t')
    print('Number of features (columns) is:', df.shape[1], sep='\t')

    categorical_dict, object_cols, noncategorical_cols = {}, [], []
    # Set display options for pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    for column in df.columns:
        categorical = stat_categorical(df[column])
        if categorical:
            categorical_dict[column] = categorical
        elif df[column].dtype == 'object':
            object_cols.append(column)
        else:
            noncategorical_cols.append(column)

    print(f"{list(categorical_dict.keys())} are likely categorical.")
    for cat in categorical_dict.items():
        print(f'{cat[0]} column has \
{cat[1]["unique_values"]} unique values out of total \
{cat[1]["total_values"]} - ({round(cat[1]["ratio"], 4)})')

    print(f"{noncategorical_cols} are not categorical.")

    noncat_stat_dict = {}

    iqr_threshold = 1.5

    for noncat in noncategorical_cols:
        try:
            noncat_stat_dict[noncat] = stat_noncategorical(df[noncat])
            print('\nStatistics for column ', noncat, ': ', sep='')
            print('Min:', noncat_stat_dict[noncat]['min'], sep='\t')
            print('Max:', noncat_stat_dict[noncat]['max'], sep='\t')
            print('Mean:', noncat_stat_dict[noncat]['mean'], sep='\t')
            print('Std:', noncat_stat_dict[noncat]['std'], sep='\t')
            print('0.25 Quartile:', noncat_stat_dict[noncat]['q25'], sep='\t')
            print('Median:', noncat_stat_dict[noncat]['median'], sep='\t')
            print('0.75 Quartile:', noncat_stat_dict[noncat]['q75'], sep='\t')
            iqr = noncat_stat_dict[noncat]['q75'] - \
                noncat_stat_dict[noncat]['q25']
            print('IQR:', iqr, sep='\t')
            outliers = ((df[noncat] < noncat_stat_dict[noncat]['q25'] - iqr_threshold * iqr) |
                        (df[noncat] > noncat_stat_dict[noncat]['q75'] + iqr_threshold * iqr))
            if not outliers.empty and len(outliers) <= 10:
                print(f'Outliers: {len(outliers)}',
                      df.loc[outliers].to_string(index=False),
                      sep='\n')
            elif len(outliers) > 10:
                print(f'Outliers: {len(outliers)}')
            else:
                print('There are no outliers according to the ± 1.5 IQR rule.')
        except (TypeError, ValueError) as e:
            print(e)
    if object_cols:
        print(object_cols, 'are columns of dtype \'object\'.')

    print('There are', df.isna().sum().sum(),
          'NAs in the dataframe in total.')

    rows_with_na = df.isnull().any(axis=1).sum()
    print(rows_with_na, 'rows with NAs in the dataframe')

    cols_with_na = df.columns[df.isna().any()].tolist()
    print('Columns with NAs:', cols_with_na)

    duplicate_rows = df.duplicated().sum()
    print('There are', duplicate_rows,
          'duplicate rows in the dataframe.')

    duplicative_rows = df.any(axis=1).duplicated().sum()
    if duplicative_rows:
        print('But', duplicative_rows, 'rows with duplicative values')
