import numpy as np
from scipy import signal as scipy_signal
from scipy.optimize import curve_fit


def moving_average(signal, window_size):
    """Скользящее среднее"""
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')


def median_filter(signal, window_size):
    """Медианный фильтр"""
    return scipy_signal.medfilt(signal, kernel_size=window_size)


def sine_function(t, A, f, phi):
    """Функция синуса для подгонки"""
    return A * np.sin(2 * np.pi * f * t + phi)


def analyze_signal(times, values):
    """Анализ сигнала и определение параметров"""

    # Применяю несколько методов фильтрации для лучшего результата
    window_size = min(15, len(values) // 10) # Адаптивное окно

    # 1. Скользящее среднее
    smoothed_ma = moving_average(values, window_size)

    # 2. Медианный фильтр
    smoothed_median = median_filter(values, window_size)

    # 3. Комбинированный подход
    combined_smooth = moving_average(smoothed_median, window_size)

    smoothed_signal = combined_smooth

    # МЕТОД 1: Нахождение амплитуды через пики сглаженного сигнала
    peaks, _ = scipy_signal.find_peaks(np.abs(smoothed_signal))
    if len(peaks) > 0:
        amplitude1 = np.median(np.abs(smoothed_signal[peaks]))
    else:
        amplitude1 = np.max(np.abs(smoothed_signal))

    # МЕТОД 2: Амплитуда как половина размаха
    amplitude2 = (np.max(smoothed_signal) - np.min(smoothed_signal)) / 2

    # Беру среднее значение, для надежности
    amplitude = (amplitude1 + amplitude2) / 2

    # Анализ частоты с помощью БПФ
    sampling_rate = 1 / (times[1] - times[0])  # Частота дискретизации
    n = len(values)

    # БПФ
    fft_result = np.fft.fft(values)
    frequencies = np.fft.fftfreq(n, 1 / sampling_rate)

    # Поиск основной частоты (игнорируем отрицательные частоты и постоянную составляющую)
    positive_freq_idx = np.where((frequencies > 0) & (frequencies < sampling_rate / 2))[0]
    main_freq_idx = np.argmax(np.abs(fft_result[positive_freq_idx]))
    main_frequency = frequencies[positive_freq_idx[main_freq_idx]]

    # Период = 1 / частота
    period = 1 / main_frequency

    # Уточняем параметры с помощью нелинейной подгонки
    try:
        # Начальные приближения для подгонки
        initial_guess = [amplitude, main_frequency, 0]  # A, f, phi

        # Подгонка кривой. Метод наименьших квадратов подбирает
        # параметры синуса, чтобы он максимально точно описывал данные
        popt, pcov = curve_fit(sine_function, times, values, p0=initial_guess, maxfev=5000)

        A_fit, f_fit, phi_fit = popt
        period_fit = 1 / f_fit

        # Используем уточненные значения
        amplitude = A_fit
        period = period_fit

    except Exception as e:
        print(f"Подгонка не удалась, используем БПФ оценку: {e}")

    """Оценка шума"""
    # 1. Созданик модели сигнала с подобранными параметрами
    model_signal = sine_function(times, amplitude, 1 / period, 0)

    # 2. Вычисляю остатки
    residuals = values - model_signal

    # 3. Оценка шума несколькими методами
    noise_std1 = np.std(residuals)

    # 4. Метод: использование высокочастотных компонентов (разности соседних точек)
    # Это нужно, чтобы отделить шум от медленных изменений сигнала
    diff_signal = np.diff(values)
    noise_std2 = np.std(diff_signal) / np.sqrt(2)

    # 5. Метод: использую скользящее стандартное отклонение остатков
    residual_ma = moving_average(residuals, 5)  # Сглаживаю остатки
    high_freq_noise = residuals - residual_ma  # Выделяю высокочастотную компоненту
    noise_std3 = np.std(high_freq_noise)

    # Беру среднюю оценку
    noise_std = (noise_std1 + noise_std2 + noise_std3)/3

    return amplitude, period, noise_std


def main():
    # Чтение файла
    filename = "variant_1 (4).txt"

    try:
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Пропускаю первые две строки (заголовок)
        data_lines = lines[2:]

        times = []
        values = []

        for line in data_lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    times.append(float(parts[0]))
                    values.append(float(parts[1]))

        times = np.array(times)
        values = np.array(values)

        # Анализ сигнала
        amplitude, period, noise_std = analyze_signal(times, values)

        # Вывод результатов
        print(f"{amplitude:.1f}")
        print(f"{period:.1f}")
        print(f"{noise_std:.1f}")

    except FileNotFoundError:
        print(f"Файл {filename} не найден!")
    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")


if __name__ == "__main__":
    main()