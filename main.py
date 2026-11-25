# main.py
import os
from src.processing import DataProcessor

def generate_report():
    """
    Главная функция запуска проекта.
    Управляет всем приложением: от загрузки данных до вывода отчета.
    """
    csv_path = os.path.join('data', 'final_tickets_tenge_df.csv')
    processor = DataProcessor(csv_path=csv_path)
    
    # 1. Запуск пайплайна предобработки
    processor.run_pipeline()
    
    # Сохраняем количество пользователей ДО удаления аномалий (как в ноутбуке)
    raw_users_count = processor.df['user_id'].nunique()
    
    # 2. Создание профилей
    processor.build_user_profile(filter_anomalies=True)
    clean_users_count = len(processor.profile)
    overall_retention = processor.profile['is_two'].mean()
    
    # 3. Сбор данных по сегментам
    df_event = processor.analyze_segment_retention('event_type_main_first')
    df_device = processor.analyze_segment_retention('device_type_canonical_first')
    df_region = processor.analyze_segment_retention('region_name_first')
    df_day = processor.analyze_segment_retention('day_of_week')
    
    # Извлекаем нужные метрики для отчета (безопасно, через .iloc[0])
    mobile_share = df_device[df_device['device_type_canonical_first'] == 'mobile']['users_per_category'].iloc[0] / clean_users_count
    concert_share = df_event[df_event['event_type_main_first'] == 'концерты']['users_per_category'].iloc[0] / clean_users_count
    
    mobile_ret = df_device[df_device['device_type_canonical_first'] == 'mobile']['share'].iloc[0]
    desktop_ret = df_device[df_device['device_type_canonical_first'] == 'desktop']['share'].iloc[0]
    concert_ret = df_event[df_event['event_type_main_first'] == 'концерты']['share'].iloc[0]
    sport_ret = df_event[df_event['event_type_main_first'] == 'спорт']['share'].iloc[0]
    
    day_spread = df_day['share'].max() - df_day['share'].min()
    
    # 4. Проверка гипотез (возвращает словарь)
    hypo_results = processor.check_hypotheses()
    
    # 5. Корреляционный анализ
    # Для числа заказов (order_id_count)
    corr_orders = processor.calculate_phik_correlation(target='order_id_count', plot=False)
    days_corr_orders = corr_orders.loc['days_since_prev_mean', 'order_id_count']
    rev_corr_orders = corr_orders.loc['revenue_rub_mean', 'order_id_count']
    max_first_order_corr_orders = corr_orders.loc[['device_type_canonical_first', 'event_type_main_first', 'day_of_week', 'region_name_first'], 'order_id_count'].max()
    
    # Для сегментов (order_segment) - именно эти цифры (0.428 и 0.281) ревьюер просил вывести корректно
    corr_seg = processor.calculate_phik_correlation(target='order_segment', plot=False)
    days_corr_seg = corr_seg.loc['days_since_prev_mean', 'order_segment']
    rev_corr_seg = corr_seg.loc['revenue_rub_mean', 'order_segment']
    max_first_order_corr_seg = corr_seg.loc[['device_type_canonical_first', 'event_type_main_first', 'day_of_week', 'region_name_first'], 'order_segment'].max()

    # 6. Формирование и вывод отчета
    report = f"""
======================================================================
📊 ОБЩИЙ ВЫВОД И РЕКОМЕНДАЦИИ ПО ПРОЕКТУ
======================================================================

В рамках проекта был проведён анализ лояльности пользователей Яндекс Афиши 
для выявления факторов, влияющих на повторные покупки.

Период анализа: 01.06.2024 – 31.10.2024.
Исходная выборка: {raw_users_count:,} уникальных пользователей.
В ходе предобработки данные были очищены от дубликатов, аномально высоких 
чеков, а также от заказов с нулевой и отрицательной выручкой.

В результате для анализа профилей осталось {clean_users_count:,} пользователей, 
а скорректированная доля возвратов составила {overall_retention:.1%}.

----------------------------------------------------------------------
🔍 КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ ИССЛЕДОВАНИЯ:
----------------------------------------------------------------------
1. Возвращаемость и точки входа: 
   Большинство клиентов ({mobile_share:.0%}) совершают первую покупку с мобильных 
   устройств, а самой популярной категорией являются концерты ({concert_share:.0%}).

2. Признаки первого заказа и возврат: 
   Пользователи, начавшие с концертов, возвращаются чаще ({concert_ret:.0%}), 
   чем пришедшие на спорт ({sport_ret:.0%}), а пользователи десктопов лояльнее 
   мобильных ({desktop_ret:.0%} против {mobile_ret:.0%}).

3. География: 
   Крупнейшие регионы (Топ-10) демонстрируют долю возврата {hypo_results['top_region_share']:.1%}, 
   тогда как менее активные — {hypo_results['other_region_share']:.1%}. Размер региона 
   имеет слабую, но заметную связь с удержанием.

4. Временные характеристики: 
   День недели первой покупки не оказывает значимого влияния на возврат. 
   Разброс долей составляет всего {day_spread:.1%}.

5. Покупательское поведение: 
   Вернувшиеся пользователи в среднем приобретают более дорогие билеты. 
   По результатам Phi_k, для сегментов заказов самую сильную связь имеют средний 
   интервал между покупками ({days_corr_seg:.3f}) и средняя выручка ({rev_corr_seg:.3f}). 
   Для точного числа заказов эти показатели составляют {days_corr_orders:.3f} и {rev_corr_orders:.3f} соответственно.
   Признаки профиля первого заказа (устройство, тип, день недели) имеют крайне 
   слабую связь с удержанием (максимум {max_first_order_corr_orders:.3f} для числа заказов 
   и {max_first_order_corr_seg:.3f} для сегментов).

----------------------------------------------------------------------
💡 РЕКОМЕНДАЦИИ ЗАКАЗЧИКУ:
----------------------------------------------------------------------
1. Сфокусироваться на стимулировании второй покупки. 
   (Около {1 - overall_retention:.0%} уходят после первого заказа -> использовать промокоды, push).
   
2. Развивать мобильный канал. 
   ({mobile_share:.0%} первых заказов с телефонов -> улучшать UX, скорость загрузки).

3. Усилить работу в регионах-лидерах. 
   (Каменевский регион и Североярская область дают половину аудитории).

4. Использовать поведенческие метрики для борьбы с оттоком. 
   (Интервал между покупками — сильнейший предиктор -> триггерные письма).

5. Отказаться от таргетинга по дням недели. 
   (День недели не влияет на удержание).
======================================================================
"""
    print(report)

if __name__ == "__main__":
    generate_report()