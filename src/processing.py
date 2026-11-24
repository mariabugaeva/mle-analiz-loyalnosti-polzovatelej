from sqlalchemy import create_engine
import pandas as pd

# Загружаем библиотеки для визуализации данных
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

class DataProcessor:
    # Указываем путь по умолчанию, чтобы не было ошибки при вызове DataProcessor()
    def __init__(self, csv_path='data/final_tickets_tenge_df.csv'):
        """
        :param csv_path: путь к локальному файлу с курсами валют
        """
        self.csv_path = csv_path
        self.df = None
        self.tenge = None
        
        # Конфигурация БД
        self.db_config = {
            'user': 'praktikum_student',
            'pwd': 'Sdf4$2;d-d30pp',
            'host': 'rc1e-oa2gp4mf9egcsq5k.mdb.yandexcloud.net',
            'port': 6432,
            'db': 'data-analyst-afisha'
        }

    def load_from_sql(self):
        """1. Загрузка первоначальных данных из БД с помощью SQL-запроса."""
        connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(
            self.db_config['user'],
            self.db_config['pwd'],
            self.db_config['host'],
            self.db_config['port'],
            self.db_config['db']
        )
        engine = create_engine(connection_string)

        query = '''
        WITH set_config_precode AS (
          SELECT set_config('synchronize_seqscans', 'off', true)
        )
        SELECT user_id, 
               device_type_canonical, 
               order_id, 
               created_dt_msk AS order_dt,
               created_ts_msk AS order_ts,
               currency_code,
               revenue,
               tickets_count,
               (created_dt_msk::date - LAG(created_dt_msk, 1, NULL) OVER (PARTITION BY user_id ORDER BY created_dt_msk)::date) AS days_since_prev,
               e.event_id,
               event_name_code AS event_name,
               event_type_main,
               service_name,
               r.region_name,
               c.city_name
        FROM afisha.purchases AS p
        INNER JOIN afisha.events AS e
        ON p.event_id=e.event_id
        INNER JOIN afisha.city AS c
        ON e.city_id=c.city_id
        INNER JOIN afisha.regions AS r
        ON c.region_id=r.region_id
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
            on='order_dt',
            how='left'
        )

        # По умолчанию копируем revenue в revenue_rub
        self.df['revenue_rub'] = self.df['revenue']
        
        # Для kzt пересчитываем через курс
        mask = (self.df['currency_code'] == 'kzt')
        self.df.loc[mask, 'revenue_rub'] = (
            self.df.loc[mask, 'revenue'] * self.df.loc[mask, 'rate_rub_per_100_kzt'] / 100
        )
        print("[Конвертация] Выручка приведена к рублям (создан revenue_rub).")
        return self.df

    def cast_types(self):
        """4. Приведение типов данных."""
        for c in ['user_id', 'order_id', 'event_id']:
            self.df[c] = self.df[c].astype('string')
            
        for c in ['device_type_canonical', 'currency_code', 'event_type_main']:
            self.df[c] = self.df[c].astype('category')
            
        self.df['tickets_count'] = self.df['tickets_count'].astype('int32')
        print("[Типы] Типы данных приведены в соответствие.")
        return self.df

    def drop_duplicates(self):
        """5. Удаление неявных (содержательных) дубликатов."""
        content_cols = ['user_id', 'order_ts', 'event_id', 'revenue', 'tickets_count']
        
        df_clean = self.df.drop_duplicates(subset=content_cols, keep='first').copy()
        
        print(f"[Дубликаты] Было строк: {len(self.df)}, стало: {len(df_clean)} "
              f"(удалено: {len(self.df) - len(df_clean)})")
        self.df = df_clean
        return self.df

    def filter_outliers_and_returns(self):
        """6. Фильтрация выбросов (99 перцентиль) и удаление возвратов (<= 0)."""
        self.df = self.df[self.df['revenue_rub'] <= self.df['revenue_rub'].quantile(0.99)].reset_index(drop=True)
        print(f"[Выбросы] После фильтра 99-го перцентиля: {len(self.df)} строк.")

        self.df = self.df[self.df['revenue_rub'] > 0].reset_index(drop=True)
        print(f"[Возвраты] После удаления возвратов и нулевых заказов: {len(self.df)} строк.")
        
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
# Пример использования:
# ==========================================
if __name__ == "__main__":
    # Теперь можно вызывать без аргументов, путь возьмется по умолчанию
    processor = DataProcessor() 
    
    # Или явно указать путь, если файл лежит в другой папке
    # processor = DataProcessor(csv_path='data/final_tickets_tenge_df.csv')
    
    final_df = processor.run_pipeline()
    print(final_df.info())