import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Парсинг даних

def parse_worldometers():
    print("Починаємо парсинг даних з Worldometers...")
    url = "https://www.worldometers.info/world-population/ukraine-population/"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    tables = soup.find_all('table')

    target_table = None
    for tbl in tables:
        if 'Year' in tbl.text and 'Population' in tbl.text:
            target_table = tbl
            break

    if target_table is None:
        raise ValueError("Не вдалося знайти потрібну таблицю. Можливо, сайт тимчасово заблокував запит.")

    years = []
    populations = []

    tbody = target_table.find('tbody')
    if tbody:
        rows = tbody.find_all('tr')
    else:
        rows = target_table.find_all('tr')

    for row in rows:
        cols = row.find_all('td')
        if len(cols) > 1:
            try:
                year = int(cols[0].text.strip())
                pop = int(cols[1].text.strip().replace(',', ''))
                years.append(year)
                populations.append(pop)
            except ValueError:
                continue

    df = pd.DataFrame({'Year': years, 'Population': populations})
    df = df.sort_values(by='Year').reset_index(drop=True)

    return df


# 2. Збереження у файл
df_real = parse_worldometers()
file_name = "population_data.csv"
df_real.to_csv(file_name, index=False)
print(f"Дані успішно збережено у файл {file_name}\n")

# Переводимо дані у numpy масиви
x_real = df_real['Year'].values
y_real = df_real['Population'].values
n = len(y_real)

# 3. Аналіз реальних даних та тренд
mean_real = np.mean(y_real)
var_real = np.var(y_real, ddof=1)
std_real = np.std(y_real, ddof=1)

print("Статистичні характеристики РЕАЛЬНИХ даних:")
print(f"Математичне сподівання: {mean_real:.2f}")
print(f"Дисперсія: {var_real:.2f}")
print(f"Середньоквадратичне відхилення (СКВ): {std_real:.2f}\n")

# Апроксимація: будуємо квадратичний тренд (поліном 2-го степеня) за МНК
coefficients = np.polyfit(x_real, y_real, 2)
trend_model = np.poly1d(coefficients)
y_trend = trend_model(x_real)

# Знаходимо похибку (залишки) між реальними даними та ідеальним трендом
residuals = y_real - y_trend
mean_res = np.mean(residuals)
std_res = np.std(residuals, ddof=1)

# 4. Синтез моделі даних
# Генеруємо синтетичний шум за нормальним законом розподілу
np.random.seed(42)
synthetic_noise = np.random.normal(loc=mean_res, scale=std_res, size=n)

y_synthetic = y_trend + synthetic_noise

mean_synth = np.mean(y_synthetic)
var_synth = np.var(y_synthetic, ddof=1)
std_synth = np.std(y_synthetic, ddof=1)

print("Статистичні характеристики СИНТЕЗОВАНИХ даних:")
print(f"Математичне сподівання: {mean_synth:.2f}")
print(f"Дисперсія: {var_synth:.2f}")
print(f"Середньоквадратичне відхилення (СКВ): {std_synth:.2f}\n")

# 5. Візуалізація результатів
plt.figure(figsize=(14, 10))

# Графік 1: Реальні дані та Тренд
plt.subplot(2, 2, 1)
plt.plot(x_real, y_real, marker='o', label='Реальні дані (Населення)', color='blue')
plt.plot(x_real, y_trend, linestyle='--', color='red', label='Квадратичний тренд')
plt.title("Динаміка населення та виділений тренд")
plt.xlabel("Рік")
plt.ylabel("Населення")
plt.grid(True)
plt.legend()

# Графік 2: Синтезовані дані
plt.subplot(2, 2, 2)
plt.plot(x_real, y_synthetic, marker='s', label='Синтезовані дані', color='green')
plt.plot(x_real, y_trend, linestyle='--', color='red', label='Квадратичний тренд')
plt.title("Синтезована адитивна модель (Тренд + Нормальний шум)")
plt.xlabel("Рік")
plt.ylabel("Населення")
plt.grid(True)
plt.legend()

# Графік 3: Порівняння Реальних і Синтезованих даних
plt.subplot(2, 2, 3)
plt.plot(x_real, y_real, label='Реальні дані', color='blue', alpha=0.7, linewidth=2)
plt.plot(x_real, y_synthetic, label='Синтезовані дані', color='green', alpha=0.7, linewidth=2)
plt.title("Порівняння реальної та синтезованої вибірок")
plt.xlabel("Рік")
plt.ylabel("Населення")
plt.grid(True)
plt.legend()

# Графік 4: Гістограма розподілу синтезованого шуму
plt.subplot(2, 2, 4)
plt.hist(synthetic_noise, bins=10, facecolor="purple", alpha=0.7, edgecolor='black')
plt.title("Гістограма нормального закону розподілу похибки")
plt.xlabel("Відхилення")
plt.ylabel("Частота")
plt.grid(True)

plt.tight_layout()
plt.show()