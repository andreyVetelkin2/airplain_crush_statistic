import matplotlib.pyplot as plt

def plot_dtp(data, years, figsize=(14, 8)):
    """
    data: dict { "Название серии": [y1, y2, y3, ...] }
    years: список годов, например [2020, 2021, 2022, 2023, 2024, 2025]
    """
    plt.figure(figsize=figsize)
    plt.grid(True, alpha=0.3)

    for label, values in data.items():
        plt.plot(years, values, marker='o', linewidth=2, label=label)

    plt.xlabel("Годы")
    plt.ylabel("Значения основных показателей безопасности ДТП")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    plt.tight_layout()
    plt.show()


# Пример использования
years = [2020, 2021, 2022, 2023, 2024, 2025]

data = {
    "1. Количество ДТП из-за нарушения ПДД водителями-мужчинами": [0.95, 0.93, 0.90, 0.87, 0.84, 0.81],
    "2. Количество ДТП из-за нарушения ПДД водителями-женщинами": [0.94, 0.92, 0.89, 0.86, 0.83, 0.79],
    "14. Количество ДТП на автомобильных дорогах общего пользования": [0.65, 0.58, 0.50, 0.43, 0.35, 0.27],
    "15. Количество ДТП с пешеходами в состоянии опьянения": [0.58, 0.50, 0.44, 0.37, 0.29, 0.17]
}

# Вызов:
plot_dtp(data, years)
