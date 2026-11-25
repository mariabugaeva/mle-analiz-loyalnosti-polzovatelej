import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import phik

# Загружаем переменные окружения из файла .env
load_dotenv()

class DataProcessor:
    def __init__(self, csv_path='data/final_tickets_tenge_df.csv'):
        self.csv_path = csv_path
        self.df = None          
        self.tenge = None       
        self.profile = None     
        
        # Читаем доступы из окружения
        self.db_config = {
            'user': os.getenv('DB_USER'),
            'pwd': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT', '6432'), # 6432 как запасной вариант
            'db': os.getenv('DB_NAME')
        }

    def load_from_sql(self):
        """1. Загрузка первоначальных данных из БД с помощью SQL-запроса."""
        # ... остальной код без изменений ...

    # ==========================================
    # 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ (SQL)
    # ==========================================
    def load_from_sql(self):
        """1. Загрузка первоначальных данных из БД с помощью SQL-запроса."""
        connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(
            self.db_config['user'], self.db_config['pwd'], 
            self.db_config['host'], self.db_config['port'], self.db_config['db']
        )
        engine = create_engine(connection_string)

        query = '''
        WITH set_config_precode AS (
          SELECT set_config('synchronize_seqscans', 'off', true)
        )
        SELECT user_id, device_type_canonical, order_id, 
               created_dt_msk AS order_dt, created_ts_msk AS order_ts,
               currency_code, revenue, tickets_count,
               (created_dt_msk::date - LAG(created_dt_msk, 1, NULL) OVER (PARTITION BY user_id ORDER BY created_dt_msk)::date) AS days_since_prev,
               e.event_id, event_name_code AS event_name, event_type_main,
               service_name, r.region_name, c.city_name
        FROM afisha.purchases AS p
        INNER JOIN afisha.events AS e ON p.event_id=e.event_id
        INNER JOIN afisha.city AS c ON e.city_id=c.city_id
        INNER JOIN afisha.regions AS r ON c.region_id=r.region_id
        WHERE ((device_type_canonical = 'mobile') OR (device_type_canonical = 'desktop')) AND e.event_type_main != 'фильм'
        ORDER BY user_id ASC
        '''
        self.df = pd.read_sql_query(query, con=engine)
        print(f"[SQL] Данные успешно выгружены. Строк: {len(self.df)}")
        return self.df

    def load_rates(self):
        """2. Загрузка локального файла с курсами валют."""
        self.tenge = pd.read_csv(self.csv_path)
        self.tenge['data'] = pd.to_datetime(self.tenge['data'])
        print(f"[CSV] Курсы валют загружены из {self.csv_path}")
        return self.tenge

    def convert_currency(self):
        """3. Присоединение курсов к SQL-выгрузке и конвертация в рубли."""
        self.df = self.df.merge(
            self.tenge[['data', 'curs']].rename(columns={'data': 'order_dt', 'curs': 'rate_rub_per_100_kzt'}),
            on='order_dt', how='left'
        )
        self.df['revenue_rub'] = self.df['revenue']
        mask = (self.df['currency_code'] == 'kzt')
        self.df.loc[mask, 'revenue_rub'] = (
            self.df.loc[mask, 'revenue'] * self.df.loc[mask, 'rate_rub_per_100_kzt'] / 100
        )
        print("[Конвертация] Выручка приведена к рублям.")
        return self.df

    def cast_types(self):
        """4. Приведение типов данных."""
        for c in ['user_id', 'order_id', 'event_id']:
            self.df[c] = self.df[c].astype('string')
        for c in ['device_type_canonical', 'currency_code', 'event_type_main']:
            self.df[c] = self.df[c].astype('category')
        self.df['tickets_count'] = self.df['tickets_count'].astype('int32')
        return self.df

    def drop_duplicates(self):
        """5. Удаление неявных дубликатов."""
        content_cols = ['user_id', 'order_ts', 'event_id', 'revenue', 'tickets_count', 'device_type_canonical', 'service_name']
        df_clean = self.df.drop_duplicates(subset=content_cols, keep='first').copy()
        self.df = df_clean
        return self.df

    def filter_outliers_and_returns(self):
        """6. Фильтрация выбросов (99 перцентиль) и удаление возвратов."""
        self.df = self.df[self.df['revenue_rub'] <= self.df['revenue_rub'].quantile(0.99)].reset_index(drop=True)
        self.df = self.df[self.df['revenue_rub'] > 0].reset_index(drop=True)
        print(f"[Очистка] Итоговое число строк: {len(self.df)}")
        return self.df

    def run_pipeline(self):
        """Запуск полного пайплайна обработки."""
        print("--- Старт обработки данных ---")
        self.load_from_sql()            
        self.load_rates()               
        self.convert_currency()         
        self.cast_types()               
        self.drop_duplicates()          
        self.filter_outliers_and_returns() 
        print("--- Обработка успешно завершена ---\n")
        return self.df


    # ==========================================
    # 2. СОЗДАНИЕ ПРОФИЛЯ ПОЛЬЗОВАТЕЛЯ (Задача 3.1)
    # ==========================================
    def build_user_profile(self, filter_anomalies=True):
        """
        Создает агрегированный профиль пользователя.
        :param filter_anomalies: если True, отфильтровывает аномалии по 99-му перцентилю количества заказов.
        """
        df_sorted = self.df.sort_values(by=['user_id', 'order_ts']).copy()
        
        # Агрегация по пользователю
        profile = df_sorted.groupby('user_id').agg(
            order_dt_min=('order_dt', 'min'),
            order_dt_max=('order_dt', 'max'),
            device_type_canonical_first=('device_type_canonical', 'first'),
            region_name_first=('region_name', 'first'),
            event_type_main_first=('event_type_main', 'first'),
            order_id_count=('order_id', 'count'),
            revenue_rub_mean=('revenue_rub', 'mean'),
            days_since_prev_mean=('days_since_prev', 'mean')
        ).reset_index()
        
        # Бинарный признак
        profile['is_two'] = (profile['order_id_count'] >= 2).astype(int)
        
        # Добавляем день недели первого заказа (для Задачи 4.3)
        profile['day_of_week'] = profile['order_dt_min'].dt.day_name()
        
        # Фильтрация аномалий по 99 перцентилю (как в ноутбуке)
        if filter_anomalies:
            row_before = len(profile)
            p99 = profile['order_id_count'].quantile(0.99)
            profile = profile[profile['order_id_count'] <= p99].reset_index(drop=True)
            print(f"[Профиль] Удалено пользователей-аномалий (> {p99} заказов): {row_before - len(profile)}")
        
        self.profile = profile
        print(f"[Профиль] Построен профиль для {len(self.profile)} пользователей.")
        return self.profile


    # ==========================================
    # 3. RETENTION ПО СЕГМЕНТАМ И ГИПОТЕЗЫ (Задачи 4.1.1 - 4.1.3)
    # ==========================================
    def analyze_segment_retention(self, segment_col):
        """
        Считает долю возврата (is_two) в разрезе указанного сегмента.
        :param segment_col: колонка профиля (например, 'event_type_main_first', 'device_type_canonical_first', 'region_name_first')
        """
        if self.profile is None:
            raise ValueError("Сначала вызовите build_user_profile()")
            
        df_seg = self.profile.groupby(segment_col).agg(
            users_per_category=('user_id', 'count'),
            sum_is_two=('is_two', 'sum')
        ).reset_index()
        
        df_seg['share'] = df_seg['sum_is_two'] / df_seg['users_per_category']
        return df_seg.sort_values(by='share', ascending=False).reset_index(drop=True)

    def check_hypotheses(self):
        """Проверка Гипотезы 1 (Спорт vs Концерты) и Гипотезы 2 (Топ-10 регионов vs остальные)."""
        if self.profile is None:
            raise ValueError("Сначала вызовите build_user_profile()")
            
        print("\n--- ПРОВЕРКА ГИПОТЕЗ ---")
        
        # Гипотеза 1
        sport_share = self.profile[self.profile['event_type_main_first'] == 'спорт']['is_two'].mean()
        concert_share = self.profile[self.profile['event_type_main_first'] == 'концерты']['is_two'].mean()
        print(f"Гипотеза 1 (Спорт vs Концерты):")
        print(f"  - Доля возврата (спорт): {sport_share:.2%}")
        print(f"  - Доля возврата (концерты): {concert_share:.2%}")
        
        # Гипотеза 2
        region_stats = self.analyze_segment_retention('region_name_first')
        region_stats = region_stats.sort_values(by='users_per_category', ascending=False).reset_index(drop=True)
        
        top_10 = region_stats.head(10)
        others = region_stats.iloc[10:]
        others = others[others['users_per_category'] >= 10] # Исключаем регионы с <10 юзерами
        
        top_share = top_10['sum_is_two'].sum() / top_10['users_per_category'].sum()
        other_share = others['sum_is_two'].sum() / others['users_per_category'].sum()
        
        print(f"Гипотеза 2 (Крупные регионы vs Менее активные):")
        print(f"  - Доля возврата в Топ-10 регионах: {top_share:.2%}")
        print(f"  - Доля возврата в остальных (>=10 юзеров): {other_share:.2%}")
        
        return top_share, other_share


    # ==========================================
    # 4. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ (Задача 4.4.1)
    # ==========================================
    def calculate_phik_correlation(self, target='order_id_count', plot=True):
        """
        Рассчитывает матрицу phi_k для выявления связей с количеством заказов.
        :param target: 'order_id_count' (число заказов) или 'order_segment' (сегменты)
        :param plot: отрисовывать ли тепловую карту
        """
        if self.profile is None:
            raise ValueError("Сначала вызовите build_user_profile()")
            
        df_corr = self.profile.copy()
        
        # Если цель - сегменты, создаем колонку order_segment
        if target == 'order_segment':
            def get_order_segment(x):
                if x == 1: return '1 заказ'
                elif x == 2: return '2 заказа'
                elif 3 <= x <= 4: return '3-4 заказа'
                else: return '5+ заказов'
                
            df_corr['order_segment'] = df_corr['order_id_count'].apply(get_order_segment)
            df_corr['order_segment'] = pd.Categorical(
                df_corr['order_segment'], 
                categories=['1 заказ', '2 заказа', '3-4 заказа', '5+ заказов'], 
                ordered=True
            )
            features = ['device_type_canonical_first', 'region_name_first', 'event_type_main_first', 
                        'day_of_week', 'revenue_rub_mean', 'days_since_prev_mean', 'order_segment']
            interval_cols = ['revenue_rub_mean', 'days_since_prev_mean']
        else:
            features = ['device_type_canonical_first', 'region_name_first', 'event_type_main_first', 
                        'is_two', 'day_of_week', 'revenue_rub_mean', 'days_since_prev_mean', 'order_id_count']
            interval_cols = ['revenue_rub_mean', 'days_since_prev_mean', 'order_id_count']

        # Считаем матрицу Phi_k
        phik_corr = df_corr[features].phik_matrix(interval_cols=interval_cols)
        
        # Оставляем только корреляцию с целевым признаком
        target_corr = phik_corr.loc[phik_corr.index != target][[target]].sort_values(by=target, ascending=False)
        print(f"\nСвязь признаков с {target}:\n{target_corr.round(3)}")
