import pandas as pd
import numpy as np
import pandas as pd
import io


def Facu_ETL(input_bucket, output_bucket):

    reviews_blob = input_bucket.blob(
        'olist_order_reviews_dataset.csv').download_as_string()
    orders_blob = input_bucket.blob(
        'olist_orders_dataset.csv').download_as_string()
    customers_blob = input_bucket.blob(
        'olist_customers_dataset.csv').download_as_string()
    state_name_blob = input_bucket.blob(
        'br-state-codes.csv').download_as_string()
    products_blob = input_bucket.blob(
        'olist_products_dataset.csv').download_as_string()
    translation_blob = input_bucket.blob(
        'product_category_name_translation.csv').download_as_string()
    payment_blob = input_bucket.blob(
        'olist_order_payments_dataset.csv').download_as_string()
    items_blob = input_bucket.blob(
        'olist_order_items_dataset.csv').download_as_string()

    olist_reviews = pd.read_csv(io.StringIO(reviews_blob.decode()))
    olist_orders = pd.read_csv(io.StringIO(orders_blob.decode()))
    olist_customers = pd.read_csv(io.StringIO(customers_blob.decode()))
    brazil_state_name = pd.read_csv(io.StringIO(state_name_blob.decode()))
    # En este bloque de código unifico el dataset orders con customers, después transformo las columas que contienen fechas en formato datetime, cambio a numerico los
    # distintos tipos de order status para que pueda ser más fácil realizar ML

    olist_orders_and_customers = olist_orders.merge(olist_customers)
    olist_orders_and_customers.iloc[:, [3, 4, 5, 6, 7]] = olist_orders_and_customers.iloc[:, [
        3, 4, 5, 6, 7]].apply(pd.to_datetime, errors='coerce')
    olist_orders_and_customers['Arrival time (In Days)'] = (
        olist_orders_and_customers.order_delivered_customer_date - olist_orders_and_customers.order_purchase_timestamp).dt.days

    conditions = [
        (olist_orders_and_customers['order_status'] == 'delivered'),
        (olist_orders_and_customers['order_status'] == 'shipped'),
        (olist_orders_and_customers['order_status'] == 'canceled'),
        (olist_orders_and_customers['order_status'] == 'unavailable'),
        (olist_orders_and_customers['order_status'] == 'invoiced'),
        (olist_orders_and_customers['order_status'] == 'processing'),
        (olist_orders_and_customers['order_status'] == 'created'),
        (olist_orders_and_customers['order_status'] == 'approved')]

    values = [1, 2, 3, 4, 5, 6, 7, 8]
    olist_orders_and_customers['numeric_order_status'] = np.select(
        conditions, values)

    # Después, junto el dataset anterior con el de reviews, después agrego un extra interesante: los nombres de los state de brazil en una columna aparte. para esto
    # hago un merge con un dataset de brazil. despues unifico todo con reviews.

    olist_orders_customers_reviews = olist_orders_and_customers.merge(
        olist_reviews)
    brazil_state_name = brazil_state_name.iloc[:27, [0, 3]]
    brazil_state_name.rename(
        columns={'subdivision': 'customer_state'}, inplace=True)
    olist_orders_and_customers = olist_orders_and_customers.merge(
        brazil_state_name)
    olist_orders_customers_reviews = olist_orders_and_customers.merge(
        olist_reviews)

    # un pequeño groupby del promedio de tiempo que tardan en llegar los productor por ciudad

    mean_arrival_review_by_customer_city = olist_orders_customers_reviews.groupby(
        by='customer_city').mean().iloc[:, [1, 3]]
    mean_arrival_review_by_customer_city.sort_values(
        by='Arrival time (In Days)', ascending=False)
    mean_arrival_review_by_customer_city_not_nan = mean_arrival_review_by_customer_city.dropna()
    mean_arrival_review_by_customer_city_not_nan.sort_values(
        by='Arrival time (In Days)', ascending=False)

    # un pequeño groupby del promedio de tiempo que tardan en llegar los productor por provincia

    mean_arrival_review_by_customer_state = olist_orders_customers_reviews.groupby(
        by='name').mean().iloc[:, [1, 3]]
    mean_arrival_review_by_customer_state.sort_values(
        by='Arrival time (In Days)', ascending=False)

    # Segundo ETL que hice, esta vez de los productos y su traducción al inglés.
    olist_products = pd.read_csv(io.StringIO(products_blob.decode()))
    olist_products_translation = pd.read_csv(
        io.StringIO(translation_blob.decode()))

    # uno los productos con su correspondiente traduccion de la subcategoria
    olist_products_whit_translations = olist_products.merge(
        olist_products_translation)

    # trabajo hecho por maga, que plantea crear categorias mas grandes o "categorias principales" por sobre las planteadas. Después de hacer esto deberían diferenciarse
    # entre categorias principales y secundarias

    beauty = ['perfumaria', 'beleza_saude', 'fraldas_higiene']

    cars = ['automotivo']

    construction = ['climatizacao', 'construcao_ferramentas_seguranca', 'casa_construcao', 'construcao_ferramentas_construcao',
                    'construcao_ferramentas_ferramentas', 'sinalizacao_e_seguranca', 'construcao_ferramentas_iluminacao',
                    'construcao_ferramentas_jardim']

    electronics = ['pcs', 'informatica_acessorios', 'consoles_games', 'tablets_impressao_imagem',  'pc_gamer', 'eletronicos', 'telefonia',
                   'telefonia_fixa',  'eletrodomesticos_2', 'dvds_blu_ray', 'portateis_cozinha_e_preparadores_de_alimentos',
                   'portateis_casa_forno_e_cafe', 'audio',  'cine_foto', 'eletroportateis']

    entertainment = ['artes',  'instrumentos_musicais', 'artes_e_artesanato', 'livros_interesse_geral',  'musica', 'livros_importados',
                     'cds_dvds_musicais', 'livros_tecnicos']

    fashion = ['fashion_calcados', 'fashion_bolsas_e_acessorios', 'fashion_underwear_e_moda_praia', 'fashion_roupa_masculina',
               'fashion_roupa_feminina',  'malas_acessorios', 'relogios_presentes', 'fashion_roupa_infanto_juvenil']

    food = ['bebidas', 'alimentos_bebidas', 'alimentos']

    home = ['utilidades_domesticas',  'moveis_decoracao', 'eletrodomesticos',  'ferramentas_jardim', 'moveis_escritorio',
            'cama_mesa_banho',  'moveis_quarto',  'moveis_cozinha_area_de_servico_jantar_e_jardim',  'moveis_sala',  'la_cuisine',
            'casa_conforto', 'moveis_colchao_e_estofado', 'flores', 'casa_conforto_2',  'papelaria', 'artigos_de_festas', 'pet_shop',
            'artigos_de_natal', 'cool_stuff']

    kids = ['bebes', 'brinquedos']

    services = ['agro_industria_e_comercio', 'seguros_e_servicos',
                'industria_comercio_e_negocios', 'market_place']

    sports = ['esporte_lazer',  'fashion_esporte']

    # creacion de la columna condicional, en la cual se analiza si la categoria secundaria pertenece a alguna categoria principal o no

    conditions = [
        (olist_products_whit_translations['product_category_name'].isin(
            beauty)),
        (olist_products_whit_translations['product_category_name'].isin(
            electronics)),
        (olist_products_whit_translations['product_category_name'].isin(cars)),
        (olist_products_whit_translations['product_category_name'].isin(
            construction)),
        (olist_products_whit_translations['product_category_name'].isin(
            entertainment)),
        (olist_products_whit_translations['product_category_name'].isin(
            fashion)),
        (olist_products_whit_translations['product_category_name'].isin(food)),
        (olist_products_whit_translations['product_category_name'].isin(home)),
        (olist_products_whit_translations['product_category_name'].isin(kids)),
        (olist_products_whit_translations['product_category_name'].isin(
            services)),
        (olist_products_whit_translations['product_category_name'].isin(sports))]

    values = ['beauty', 'electronics', 'cars', 'construction',
              'entertainment', 'fashion', 'food', 'home', 'kids', 'services', 'sports']
    olist_products_whit_translations['product_principal_category'] = np.select(
        conditions, values)

    # carga del dataset de payments, que puede añadirse a lo anteriormente mencionado

    olist_payments = pd.read_csv(io.StringIO(payment_blob.decode()))

    # como dije anteriormente, se une todos los dataset y se pueden comenzar a trabajar sin problema.

    olist_order_items = pd.read_csv(io.StringIO(items_blob.decode()))
    list_order_products_payments = olist_order_items.merge(
        olist_products_whit_translations).merge(olist_payments)

    # esto esta hecho solamente para poder trabajar con este dataset en powerbi, quitando las columnas de titulo y mensaje de review ya que no tienen utilidad para data analytics
    # sin el nlp

    olist_orders_customers_reviews_pbi = olist_orders_customers_reviews.drop(
        columns=['review_comment_title', 'review_comment_message'])

    # guardado de datasets

    data = olist_orders_customers_reviews_pbi.to_csv(index=False)

    output_bucket.blob(
        'orders_customers_reviews_pbi.csv').upload_from_string(data)

    data = list_order_products_payments.to_csv(index=False)
    output_bucket.blob('order_products_payments.csv').upload_from_string(data)


def etl_orders_payment(input_bucket, output_bucket):

    orders_blob = input_bucket.blob(
        'olist_orders_dataset.csv').download_as_string()
    payment_blob = input_bucket.blob(
        'olist_order_payments_dataset.csv').download_as_string()
    closed_deals = input_bucket.blob(
        'olist_closed_deals_dataset.csv').download_as_string()

    orders = pd.read_csv(io.StringIO(orders_blob.decode()), usecols=[
                         'customer_id', 'order_id', 'order_purchase_timestamp'])
    payment = pd.read_csv(io.StringIO(payment_blob.decode()), usecols=[
                          'order_id', 'payment_value'])
    closed_deals = pd.read_csv(io.StringIO(closed_deals.decode()))

    orders_payment = orders.merge(payment, on='order_id', how='inner')
    orders_payment = orders_payment[[
        'customer_id', 'order_purchase_timestamp', 'order_id', 'payment_value']]
    orders_payment.rename(columns={'payment_value': 'order_value',
                          'order_purchase_timestamp': 'order_date'}, inplace=True)
    orders_payment['order_date'] = orders_payment['order_date'].astype(
        'datetime64[ns]')
    orders_payment['month_yr'] = orders_payment['order_date'].apply(
        lambda x: x.strftime('%b-%Y'))

    data = orders_payment.to_csv(index=False)

    output_bucket.blob('orders_payment.csv').upload_from_string(data)


def etl_qualified_leads(input_bucket, output_bucket):

    mql_blob = input_bucket.blob(
        'olist_marketing_qualified_leads_dataset.csv').download_as_string()
    closed_deals_blob = input_bucket.blob(
        'olist_closed_deals_dataset.csv').download_as_string()

    mql = pd.read_csv(io.StringIO(mql_blob.decode()), usecols=[
                      'mql_id', 'landing_page_id', 'origin'])
    closed_deals = pd.read_csv(io.StringIO(closed_deals_blob.decode()), usecols=[
                               'mql_id', 'seller_id', 'business_segment', 'lead_type', 'lead_behaviour_profile', 'business_type'])
    qualified_leads = mql.merge(closed_deals, how='inner')
    qualified_leads = qualified_leads[['landing_page_id', 'origin', 'seller_id',
                                       'business_segment', 'lead_type', 'lead_behaviour_profile', 'business_type']]

    data = qualified_leads.to_csv(index=False)

    output_bucket.blob('etl_qualified_leads.csv').upload_from_string(data)
    input_bucket.blob('etl_qualified_leads.csv').upload_from_string(data)


def etl_cltv(input_bucket, output_bucket):

    ql_blob = input_bucket.blob('etl_qualified_leads.csv').download_as_string()
    sellers_blob = input_bucket.blob(
        'olist_sellers_dataset.csv').download_as_string()
    items_blob = input_bucket.blob(
        'olist_order_items_dataset.csv').download_as_string()

    qualified_leads = pd.read_csv(io.StringIO(ql_blob.decode()))
    sellers = pd.read_csv(io.StringIO(sellers_blob.decode()))
    items = pd.read_csv(io.StringIO(items_blob.decode()))
    items = items[['order_id',
                   # 'order_item_id',
                   # 'product_id',
                   'seller_id',
                   # 'shipping_limit_date',
                   'price',
                   # 'freight_value'
                   ]]

    seller_sales = items.groupby(by='seller_id')['price'].sum()
    seller_sales = seller_sales.reset_index()
    seller_sales.columns = ['seller_id', 'seller_sales']

    # CLTV general
    cltv = seller_sales.merge(sellers)
    cltv.rename(columns={'seller_sales': 'cltv'}, inplace=True)
    cltv = cltv[['seller_id', 'seller_zip_code_prefix',
                 'seller_city', 'seller_state', 'cltv']]
    data = cltv.to_csv(index=False)
    output_bucket.blob('etl_cltv.csv').upload_from_string(data)

    # CLTV para Qualified Marketing Leads
    # Aquí iría el costo del paid_search, consultar al área de mkt
    paid_search_price = 0
    qml_cltv = qualified_leads.merge(seller_sales)
    qml_cltv[qml_cltv['origin'] ==
             'paid_search']['seller_sales'] -= paid_search_price
    qml_cltv.rename(columns={'seller_sales': 'cltv'}, inplace=True)

    data = qml_cltv.to_csv(index=False)
    output_bucket.blob('etl_qml_cltv.csv').upload_from_string(data)


def etl_closed_deals(input_bucket, output_bucket):

    closed_deal_blob = input_bucket.blob(
        'olist_closed_deals_dataset.csv').download_as_string()

    closed_deals = pd.read_csv(io.StringIO(closed_deal_blob.decode()), usecols=[
                               'mql_id', 'seller_id', 'sdr_id', 'sr_id', 'won_date', 'business_segment', 'lead_type', 'lead_behaviour_profile', 'business_type', 'declared_monthly_revenue'])

    data = closed_deals.to_csv(index=False)
    output_bucket.blob('closed_deals_dataset.csv').upload_from_string(data)


def etl_PODTCWTLM(input_bucket, output_bucket):
    order_customers_review_pbi = input_bucket.blob(
        'orders_customers_reviews_pbi.csv').download_as_string()

    order_customers_review_pbi = pd.read_csv(
        io.StringIO(order_customers_review_pbi.decode()))

    selection = order_customers_review_pbi.iloc[:, [2, 3, 4, 5, 6, 7, 12]]

    selection_date = selection.iloc[:, [1, 2, 3, 4, 5]].apply(
        pd.to_datetime, errors='coerce')
    selection_date['order_status'] = selection['order_status']
    selection_date['Arrival time (In Days)'] = selection['Arrival time (In Days)']

    selection_date['estimated_time'] = (
        selection_date.order_estimated_delivery_date - selection_date.order_approved_at).dt.days
    selection_date['KPI'] = np.where(
        selection_date['Arrival time (In Days)'] < selection_date['estimated_time'], 'at time', 'no')

    selection_date['year'] = pd.DatetimeIndex(
        selection_date['order_approved_at']).year
    selection_date['month'] = pd.DatetimeIndex(
        selection_date['order_delivered_customer_date']).month

    selection_date['Arrival time (In Days)'] = selection_date['Arrival time (In Days)'].values.astype(
        int)
    selection_date['year'] = selection_date['year'].values.astype(int)
    selection_date['month'] = selection_date['month'].values.astype(int)
    selection_date['estimated_time'] = selection_date['estimated_time'].values.astype(
        int)
    selection_date['at time'] = pd.get_dummies(
        selection_date['KPI']).iloc[:, 0]
    selection_date['no at time'] = pd.get_dummies(
        selection_date['KPI']).iloc[:, 1]
    selection_date.dropna(inplace=True)

    siuuu = round((selection_date.groupby([selection_date['year'].rename(
        'year'), selection_date['month'].rename('month')]).mean()*100), 2)
    finished = siuuu.loc[:, ['at time', 'no at time']]
    finished_reset_index = finished.reset_index()
    finished_reset_index.year = finished_reset_index.year.astype(str)
    finished_reset_index.month = finished_reset_index.month.astype(str)
    finished_reset_index['first'] = pd.to_datetime(
        finished_reset_index['month'] + '/' + finished_reset_index['year'].astype(str), format='%m/%Y')
    finished_reset_index['last'] = finished_reset_index['first'] + \
        pd.offsets.MonthEnd(n=1)

    data = finished_reset_index.to_csv(index=False)
    output_bucket.blob('PODTCWTLM.csv').upload_from_string(data)


def etl_CGRATMCCTTPM(input_bucket, output_bucket):
    leads = input_bucket.blob(
        'olist_marketing_qualified_leads_dataset.csv').download_as_string()
    leads = pd.read_csv(io.StringIO(leads.decode()))

    leads['first_contact_date'] = leads['first_contact_date'].apply(
        pd.to_datetime, errors='coerce')
    leads['year'] = pd.DatetimeIndex(leads['first_contact_date']).year
    leads['month'] = pd.DatetimeIndex(leads['first_contact_date']).month
    leads_frame = leads.groupby([leads['year'].rename('year'), leads['month'].rename(
        'month'), leads['origin']])['origin'].count().to_frame()
    leads_finished = leads_frame.rename(
        columns={'origin': 'count'}).reset_index()
    leads_finished.year = leads_finished.year.astype(str)
    leads_finished.month = leads_finished.month.astype(str)
    leads_finished['first'] = pd.to_datetime(
        leads_finished['month'] + '/' + leads_finished['year'].astype(str), format='%m/%Y')
    leads_finished['last'] = leads_finished['first'] + pd.offsets.MonthEnd(n=1)

    data = leads_finished.to_csv(index=False)

    output_bucket.blob('CGRATMCCTTPM.csv').upload_from_string(data)

    leads_count = leads.origin.value_counts().to_frame().reset_index()
    leads_count.rename(columns={'origin': 'count'}, inplace=True)
    leads_count.rename(columns={'index': 'origin'}, inplace=True)

    data = leads_count.to_csv(index=False)
    output_bucket.blob('CARBO.csv').upload_from_string(data)


def etl_geolocation(input_bucket, output_bucket):
    geolocalization = input_bucket.blob(
        'olist_geolocation_dataset.csv').download_as_string()
    geolocalization = pd.read_csv(io.StringIO(geolocalization.decode()))

    geolocalization.geolocation_lat = geolocalization.geolocation_lat.astype(
        str)
    geolocalization.geolocation_lng = geolocalization.geolocation_lng.astype(
        str)
    geolocalization['location'] = geolocalization['geolocation_lat'] + \
        ','+geolocalization['geolocation_lng']

    geolocalization.to_csv('geolocation.csv', index=False)
    # output_bucket.blob('geolocation.csv').upload_from_string(data)


def etl_MAPOPBCWLM(input_bucket, output_bucket):
    pbi = input_bucket.blob(
        'orders_customers_reviews_pbi.csv').download_as_string()
    products = input_bucket.blob(
        'order_products_payments.csv').download_as_string()

    pbi = pd.read_csv(io.StringIO(pbi.decode()))
    products = pd.read_csv(io.StringIO(products.decode()))

    last_kpi = pbi.merge(products)
    last_kpi = last_kpi.iloc[:, [0, 20, 4, 14]]
    last_kpi['order_approved_at'] = last_kpi['order_approved_at'].apply(
        pd.to_datetime, errors='coerce')
    last_kpi['year'] = pd.DatetimeIndex(last_kpi['order_approved_at']).year
    last_kpi['month'] = pd.DatetimeIndex(last_kpi['order_approved_at']).month
    last_kpi['year'] = last_kpi['year'].values.astype(int)
    last_kpi['month'] = last_kpi['month'].values.astype(int)
    last_kpi = last_kpi.groupby(
        [last_kpi['order_id'], last_kpi['year'], last_kpi['month']]).count()
    last_kpi = last_kpi.reset_index()
    last_kpi.drop(columns='name', inplace=True)
    last_kpi.drop(columns='order_approved_at', inplace=True)
    last_kpi = last_kpi.loc[last_kpi['year'] > 1]
    last_kpi.year = last_kpi.year.astype(str)
    last_kpi.month = last_kpi.month.astype(str)
    last_kpi['first'] = pd.to_datetime(
        last_kpi['month'] + '/' + last_kpi['year'].astype(str), format='%m/%Y')
    last_kpi['last'] = last_kpi['first'] + pd.offsets.MonthEnd(n=1)

    data = last_kpi.to_csv(index=False)
    output_bucket.blob('MAPOPBCWLM.csv').upload_from_string(data)
