import streamlit as st
import joblib
import pandas as pd
st.title('🚗 CAR PRICE PREDICTION 🚗')

# Từ điển các mô hình xe
car_models_dict = {
'Hyundai': ['Hyundai Santro Xing', 'Hyundai Grand i10', 'Hyundai Eon', 'Hyundai Elite i20', 'Hyundai i20 Sportz',
'Hyundai i20 Magna', 'Hyundai i20 Asta', 'Hyundai Verna', 'Hyundai Creta', 'Hyundai i10 Magna',
'Hyundai Verna Fluidic', 'Hyundai i20 Active', 'Hyundai i10', 'Hyundai Accent', 'Hyundai Verna',
'Hyundai Fluidic Verna', 'Hyundai Getz Prime', 'Hyundai Santro', 'Hyundai Getz', 'Hyundai Xcent Base',
'Hyundai Accent Executive', 'Hyundai Sonata Transform', 'Hyundai Elantra SX', 'Hyundai Verna 1.4',
'Hyundai Xcent SX', 'Hyundai Eon Era', 'Hyundai Eon Magna', 'Hyundai Eon D', 'Hyundai i20 Select'],
'Mahindra': ['Mahindra Jeep CL550', 'Mahindra Scorpio SLE', 'Mahindra Scorpio S10', 'Mahindra Bolero DI',
'Mahindra XUV500 W8', 'Mahindra TUV300 T4', 'Mahindra Thar CRDe', 'Mahindra Quanto C8',
'Mahindra Scorpio S4', 'Mahindra Scorpio VLX', 'Mahindra Xylo E4', 'Mahindra Bolero SLE',
'Mahindra Xylo E8', 'Mahindra Bolero Power', 'Mahindra KUV100 K8', 'Mahindra Scorpio SLX',
'Mahindra Sumo Victa', 'Mahindra XUV500 W6', 'Mahindra Bolero DI', 'Mahindra Jeep MM',
'Mahindra Bolero DI', 'Mahindra XUV500 W10', 'Mahindra TUV300 T8', 'Mahindra Scorpio Vlx',
'Mahindra XUV500', 'Mahindra Thar CRDe', 'Mahindra Bolero Power', 'Mahindra Scorpio VLX',
'Mahindra Scorpio 2.6', 'Mahindra Logan Diesel', 'Mahindra Quanto C4', 'Mahindra Bolero SLE',
'Mahindra TUV300 T4'],
'Ford': ['Ford EcoSport Titanium', 'Ford Figo', 'Ford EcoSport Ambiente', 'Ford EcoSport', 'Ford Figo Diesel',
'Ford Figo Petrol', 'Ford EcoSport Trend', 'Ford Figo Duratorq', 'Ford Fiesta', 'Ford Fiesta SXi',
'Ford Ikon 1.3', 'Ford Ikon 1.6', 'Ford Fusion 1.4'],
'Maruti': ['Maruti Suzuki Alto', 'Maruti Suzuki Stingray', 'Maruti Suzuki Vitara', 'Maruti Suzuki Swift',
'Maruti Suzuki Wagon', 'Maruti Suzuki Dzire', 'Maruti Suzuki Baleno', 'Maruti Suzuki Ertiga',
'Maruti Suzuki Eeco', 'Maruti Suzuki Esteem', 'Maruti Suzuki Ritz', 'Maruti Suzuki Ciaz',
'Maruti Suzuki Zen', 'Maruti Suzuki 800', 'Maruti Suzuki Omni', 'Maruti Suzuki Versatile',
'Maruti Suzuki Estilo', 'Maruti Suzuki Versa', 'Maruti Suzuki S', 'Maruti Suzuki Stingray',
'Maruti Suzuki Dzire', 'Maruti Suzuki A', 'Maruti Suzuki Ertiga', 'Maruti Suzuki Ritz',
'Maruti Suzuki Swift', 'Maruti Suzuki Baleno', 'Maruti Suzuki Esteem', 'Maruti Suzuki Versa',
'Maruti Suzuki Eeco', 'Maruti Suzuki Ertiga', 'Maruti Suzuki SX4', 'Maruti Suzuki Stingray',
'Maruti Suzuki Baleno', 'Maruti Suzuki Vitara', 'Maruti Suzuki Swift', 'Maruti Suzuki Zen'],
'Skoda': ['Skoda Fabia Classic', 'Skoda Fabia 1.2L', 'Skoda Yeti Ambition', 'Skoda Rapid Elegance',
'Skoda Laura', 'Skoda Superb 1.8', 'Skoda Octavia Classic', 'Skoda Octavia', 'Skoda Octavia Ambition',
'Skoda Octavia Elegance'],
'Audi': ['Audi A8', 'Audi Q7', 'Audi A4 1.8', 'Audi A4 2.0', 'Audi A6 2.0', 'Audi Q3 2.0', 'Audi A4',
'Audi A6', 'Audi Q3', 'Audi Q7', 'Audi Q5', 'Audi A3'],
'Toyota': ['Toyota Innova 2.0', 'Toyota Corolla Altis', 'Toyota Fortuner', 'Toyota Innova 2.5', 'Toyota Corolla H2',
'Toyota Etios', 'Toyota Innova', 'Toyota Etios GD', 'Toyota Corolla', 'Toyota Qualis', 'Toyota Innova',
'Toyota Fortuner 3.0', 'Toyota Corolla', 'Toyota Etios Liva', 'Toyota Etios Cross', 'Toyota Etios',
'Toyota Camry', 'Toyota Yaris', 'Toyota Corolla Altis', 'Toyota Corolla', 'Toyota Prius'],
'Renault': ['Renault Lodgy 85', 'Renault Duster 110', 'Renault Duster 85', 'Renault Scala RxL', 'Renault Kwid RXT',
'Renault Kwid', 'Renault Duster 110PS', 'Renault Duster', 'Renault Scala', 'Renault Pulse',
'Renault Fluence', 'Renault Koleos', 'Renault Captur', 'Renault Triber', 'Renault Kwid Climber',
'Renault Kwid RXE', 'Renault Kwid RXL', 'Renault Kwid 1.0'],
'Honda': ['Honda City 1.5', 'Honda Amaze', 'Honda Amaze 1.5', 'Honda City', 'Honda Brio', 'Honda Jazz',
'Honda WR V', 'Honda Mobilio', 'Honda CR-V', 'Honda Accord', 'Honda City', 'Honda Jazz S',
'Honda City ZX', 'Honda Brio V', 'Honda Jazz VX', 'Honda Mobilio S', 'Honda Amaze 1.2',
'Honda Accord', 'Honda Brio VX', 'Honda City VX', 'Honda Jazz', 'Honda CRV'],
'Datsun': ['Datsun Redi GO', 'Datsun GO T', 'Datsun GO Plus', 'Datsun Go', 'Datsun Go Plus T', 'Datsun Redi GO T'],
'Mitsubishi': ['Mitsubishi Pajero Sport', 'Mitsubishi Lancer 1.8', 'Mitsubishi Outlander', 'Mitsubishi Montero',
'Mitsubishi Cedia', 'Mitsubishi Lancer', 'Mitsubishi Pajero', 'Mitsubishi Eclipse Cross'],
'Tata': ['Tata Indigo eCS', 'Tata Indica V2', 'Tata Indica', 'Tata Indigo', 'Tata Nano', 'Tata Manza',
'Tata Aria', 'Tata Zest', 'Tata Safari', 'Tata Hexa', 'Tata Sumo', 'Tata Bolt', 'Tata Tiago',
'Tata Nexon', 'Tata Harrier', 'Tata Altroz', 'Tata Punch', 'Tata Venture', 'Tata Safari Storme',
'Tata Tigor', 'Tata Indigo Marina', 'Tata Manza Aqua', 'Tata Sumo Grande', 'Tata Sumo Victa',
'Tata Vista Quadrajet', 'Tata Zest Quadrajet', 'Tata Tigor Revotron', 'Tata Nano GenX',
'Tata Sumo Gold', 'Tata Safari Dicor', 'Tata Xenon', 'Tata Nano Lx', 'Tata Nano LX', 'Tata Indica eV2',
'Tata Indica Vista', 'Tata Manza Elan', 'Tata Indigo XL', 'Tata Indica V2', 'Tata Indica LSi',
'Tata Indigo LX', 'Tata Indica V2 DLS', 'Tata Indica Vista Quadrajet'],
'Volkswagen': ['Volkswagen Polo Highline', 'Volkswagen Polo Comfortline', 'Volkswagen Polo', 'Volkswagen Vento',
'Volkswagen Passat', 'Volkswagen Jetta', 'Volkswagen Ameo', 'Volkswagen Beetle',
'Volkswagen Tiguan', 'Volkswagen Touareg', 'Volkswagen Polo Trendline', 'Volkswagen Vento Highline',
'Volkswagen Jetta Highline', 'Volkswagen Polo Highline1.2L', 'Volkswagen Vento Comfortline',
'Volkswagen Polo Select', 'Volkswagen Vento Konekt', 'Volkswagen Passat Diesel', 'Volkswagen Polo Select',
'Volkswagen Polo Highline', 'Volkswagen Polo Select', 'Volkswagen Polo GT', 'Volkswagen Ameo Comfortline',
'Volkswagen Polo Select', 'Volkswagen Jetta Comfortline', 'Volkswagen Polo GT TSI'],
'Chevrolet': ['Chevrolet Spark LS', 'Chevrolet Spark', 'Chevrolet Beat LT', 'Chevrolet Beat', 'Chevrolet Enjoy',
'Chevrolet Cruze', 'Chevrolet Tavera', 'Chevrolet Captiva', 'Chevrolet Sail', 'Chevrolet Trailblazer',
'Chevrolet Spark LT', 'Chevrolet Beat Diesel', 'Chevrolet Tavera LS', 'Chevrolet Beat LS',
'Chevrolet Beat PS', 'Chevrolet Tavera Neo', 'Chevrolet Spark 1.0', 'Chevrolet Enjoy 1.4',
'Chevrolet Sail UVA', 'Chevrolet Cruze LTZ', 'Chevrolet Sail 1.2', 'Chevrolet Spark LTZ',
'Chevrolet Aveo', 'Chevrolet Beat LT', 'Chevrolet Optra Magnum'],
'Mini': ['Mini Cooper S', 'Mini Cooper Countryman', 'Mini Cooper', 'Mini Cooper Convertible', 'Mini Cooper Clubman',
'Mini Cooper JCW'],
'BMW': ['BMW 3 Series', 'BMW 7 Series', 'BMW 5 Series', 'BMW X1', 'BMW X3', 'BMW X5', 'BMW 6 Series', 'BMW Z4',
'BMW 2 Series', 'BMW 1 Series', 'BMW i8', 'BMW X7', 'BMW X6', 'BMW M2', 'BMW M5', 'BMW M3', 'BMW X1 sDrive20d',
'BMW X1 xDrive20d'],
'Nissan': ['Nissan Micra XV', 'Nissan Micra', 'Nissan Sunny', 'Nissan Terrano', 'Nissan GT-R', 'Nissan X-Trail',
'Nissan Evalia', 'Nissan 370Z', 'Nissan Patrol', 'Nissan Leaf', 'Nissan Micra XL', 'Nissan Sunny XL',
'Nissan X-Trail', 'Nissan Terrano XL', 'Nissan Kicks', 'Nissan Magnite', 'Nissan Micra Active',
'Nissan Sunny Special Edition', 'Nissan Terrano Diesel', 'Nissan Micra Diesel', 'Nissan GT-R Nismo'],
'Hindustan': ['Hindustan Motors Ambassador', 'Hindustan Contessa'],
'Fiat': ['Fiat Punto Emotion', 'Fiat Linea', 'Fiat Punto', 'Fiat Avventura', 'Fiat 500', 'Fiat Abarth Punto',
'Fiat Urban Cross', 'Fiat Punto Pure', 'Fiat Linea Classic', 'Fiat Punto Evo', 'Fiat Punto Pure'],
'Force': ['Force Motors Force', 'Force Motors One', 'Force Gurkha', 'Force Traveller', 'Force Trax Cruiser'],
'Mercedes': ['Mercedes Benz B', 'Mercedes Benz C', 'Mercedes Benz A', 'Mercedes Benz E', 'Mercedes Benz S',
'Mercedes Benz GLA', 'Mercedes Benz CLA', 'Mercedes Benz GLB', 'Mercedes Benz GLC', 'Mercedes Benz GLS',
'Mercedes Benz GLE', 'Mercedes Benz G', 'Mercedes Benz AMG', 'Mercedes Benz V-Class'],
'Jeep': ['Jeep Wrangler Unlimited', 'Jeep Compass', 'Jeep Grand Cherokee', 'Jeep Renegade', 'Jeep Wrangler',
'Jeep Gladiator', 'Jeep Wagoneer'],
'Volvo': ['Volvo S80 Summum', 'Volvo XC40', 'Volvo XC60', 'Volvo XC90', 'Volvo S90', 'Volvo V90', 'Volvo V40',
'Volvo S60', 'Volvo S60 Cross Country', 'Volvo V60', 'Volvo V60 Cross Country']
}
st.header('Vui lòng nhập các đặc trưng của chiếc xe bạn muốn mua:')

company = st.selectbox('Company:', list(car_models_dict.keys()))
car_model = st.selectbox('Car model:', car_models_dict[company])
year = st.number_input('Year:', min_value=1900, max_value=3000, value=2007)
kms_driven = st.number_input("Kms driven:", min_value=0, max_value=500000, value=45000)
fuel_type = st.selectbox('Fuel type:', ['Petrol', 'Diesel', 'LPG'])

@st.cache_data
def predicts(company, car_model, year, fuel_type, kms_driven):
    try:
        # Tạo DataFrame từ đầu vào
        input_data = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                                  columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        # Tải mô hình và dự đoán giá
        model = joblib.load('car.pkl')
        prediction = model.predict(input_data)
        return prediction[0]
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        return None

if st.button('Predict Price'):
    with st.spinner('Predicting price...'):
        out = predicts(company, car_model, year, fuel_type, kms_driven)
    if out is not None:
        st.success(f'Giá dự đoán của chiếc xe đó là: ${out: .2f} USD')
