import pandas as pd
from scipy.stats import chi2


def pearson_chi_squared(col1, col2, alpha):

    print('\n Defining Hypothesis')
    print('\n Null Hypothesis: There is no association between', col1.name, 'and', col2.name)
    print('\n Alternative Hypothesis: There is an association between', col1.name, 'and', col2.name)

    table = pd.crosstab(col1, col2, margins=False)

    sub_component = 0
    e_frequencies = []
    freq_less_than_five = []
    freq_less_than_or_equal_one = []
    for i in range(0, len(table.index)):
        for j in range(0, len(table.columns)):
            expected_frequency = (table.iloc[:, j:(j + 1)].sum().sum() * table.iloc[i: i + 1, :].sum(axis=1).sum()) \
                                 / table.sum(axis=1).sum()
            e_frequencies.append(expected_frequency)
            observed_frequency = table.iloc[i, j]
            sub_component = ((observed_frequency - expected_frequency) ** 2 / expected_frequency) + sub_component
            if expected_frequency < 5:
                freq_less_than_five.append((i, j))

            elif expected_frequency <= 1:
                freq_less_than_or_equal_one.append((i, j))

    e_ratio = len(freq_less_than_five) / len(e_frequencies)
    e_len = len(freq_less_than_or_equal_one)

    test_Statistics = sub_component
    degrees_of_freedom = [(len(table.columns) - 1) * (len(table.index) - 1)]
    a = alpha / 100
    chi_critical = chi2.isf(q=a, df=degrees_of_freedom)
    p_value = chi2.sf(test_Statistics, degrees_of_freedom)

    print('\n Rejection Criteria: Reject Null Hypothesis if Test Statistic is greater than or equal to Critical Value '
          'at', alpha, '% level of Significance.')
    print('\n Test Results')
    print('\n', pd.crosstab(col1, col2, margins=True))
    results = {'Categorical Variable 1': col1.name ,'Categorical Variable 2': col2.name, 'Test Statistic': round(test_Statistics, 4),
               'Critical Value': round(chi_critical[0], 4), 'P value': p_value[0]}
    print('\n', results)

    if e_ratio <= 0.2:
        print('\n Note : No more than 20% of expected frequencies are less than 5.')

    elif e_ratio > 0.2:
        print('\n Warning : More than 20% of expected frequencies are less than 5. Hence, Chi-Squared Critical '
              'Value is invalidated by small expected frequencies.')

    elif e_len >= 1:
        print('\n Warning :', len(freq_less_than_or_equal_one),
              'number of expected frequency with less than or equal one is indicated.'
              'Hence, Chi-Squared Critical Value is invalidated by small expected frequency.')

    elif e_ratio > 0.2 and e_len >= 1:
        print('\n Warning : More than 20% of expected frequencies are less than 5 and '
              'also ', len(freq_less_than_or_equal_one),
              'number of expected frequency with less than or equal one is indicated. '
              'Hence, Chi-Squared Critical Value is invalidated by small expected frequencies.')


"""Applying Pearson's chi square test for Sepsis data"""

print("\n Results for Sepsis data")

data2 = pd.read_csv("Sepsis.csv")
column1 = data2['THERAPY']
column2 = data2['Age_Group']

pearson_chi_squared(column1, column2,alpha = 5)


"""Applying Pearson's chi square test for product and payment data"""

print("\n Results for product and payment data")

data1 = pd.read_csv("Product and Payment.csv")
column3 = data1['Type of Product']
column4 = data1['Type of Payment']

pearson_chi_squared(column3, column4,alpha = 5)

