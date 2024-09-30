
import numpy as np
import matplotlib.pyplot as plt

# Birth rates per 1000 people
india = [20.2, 19.9, 19.5, 18.9, 18.2]
nepal = [20.1, 19.7, 19.2, 18.7, 18.2]
bhutan = [16.5, 16.3, 16.1, 15.9, 15.7]

years = [2016, 2017, 2018, 2019, 2020]

# --------------------------------------------------------------------------------------------
# Bar graph of above data]
plt.figure(figsize=(12, 8))

x = np.arange(len(years))
bar_width = 0.25
plt.bar(x, india, width = bar_width, color = '#33FFFF', label = 'India')
plt.bar(x + bar_width, nepal, width = bar_width, color = '#FFD1DC', label = 'Nepal')
plt.bar(x + 2*bar_width, bhutan, width = bar_width, color = '#CDC0FF', label = 'Bhutan')

plt.legend()
plt.title('Birth Rates (2016-2020)', fontweight='bold')
plt.xlabel('Year', fontweight='bold')
plt.ylabel('Birth Rate (per 1000 people)', fontweight='bold')
plt.xticks(x + bar_width , years)

plt.show()
# --------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------
# Line chart
plt.plot(years, india, marker='o', label='India')
plt.plot(years, nepal, marker='o', label='Nepal')
plt.plot(years, bhutan, marker='o', label='Bhutan')

plt.xlabel('Year', fontweight='bold')
plt.ylabel('Birth Rate (per 1000 people)', fontweight='bold')
plt.title('Birth Rates Comparison (2016-2020)', fontweight='bold')
plt.locator_params(axis='x', integer=True)
plt.legend()

plt.grid(True)
plt.show()
# --------------------------------------------------------------------------------------------