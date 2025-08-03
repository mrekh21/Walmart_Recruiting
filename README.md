# Walmart-ის გაყიდვების პროგნოზირება

ეს პროექტი მიზნად ისახავს Walmart-ის მაღაზიების **ყოველკვირეული გაყიდვების პროგნოზირებას**. გამოყენებულია **Tree-Based Models** (LightGBM, XGBoost), **Classical Statistical Time-Series Models** (ARIMA, SARIMA, SARIMAX), **Deep Learning Models** (TFT, PatchTST, DLinear), მონაცემთა ანალიზი (EDA) და სხვადასხვა time-series/seasonal plot-ები. ექსპერიმენტები დალოგილია **MLflow**-ით და **DagsHub**-ით.

### Kaggle Competition:
https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/overview 

### Mlflow Link:
https://dagshub.com/mrekh21/Walmart_Recruiting.mlflow/#/experiments/0

### Dagshub Link:
https://dagshub.com/mrekh21/Walmart_Recruiting


# მონაცემები

პროექტში გამოყენებული მონაცემები შედგება შემდეგი ფაილებისგან:

- `train.csv.zip` – ძველი წლების/თვეების/კვირების გაყიდვების მონაცემები
- `test.csv.zip` – სატესტო მონაცემები პროგნოზისთვის
- `features.csv.zip` – ეკონომიკური და მარკეტინგული მახასიათებლები
- `stores.csv` – მაღაზიების ტიპისა და ზომის შესახებ მონაცემები

# შეფასების მეტრიკა:
```
def wmae(y_true, y_pred, is_holiday):
    weights = np.where(is_holiday == 1, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
```


# რეპოზიტორიის სტრუქტურა
```
├── Walmart_Recruiting_Data_Exploration.ipynb
├── model_experiment_XGBoost.ipynb
├── model_experiment_LightGBM.ipynb
├── model_experiment_ARIMA_SARIMA_SARIMAX.ipynb
├── model_experiment_TFT.ipynb
├── model_experiment_PatchTST.ipynb
├── model_experiment_DLinear.ipynb
├── model_inference.ipynb
├── README.md
```



# Walmart_Recruiting_Data_Exploration.ipynb ნოუთბუქის აღწერა:
- თითოეული csv ფაილის feature-ების ტიპები, გამოტოვებული მონაცემების რაოდენობა და ა.შ არის შესწავლილი
- სატრენინგო მონაცემების სხვადასხვა time-series/seasonal plot-ებია წარმოდგენილი matplotlib, seaborn, plotly, statsmodels ბიბლიოთეკების გამოყენებით
- mlflow-ზე დალოგილია შესაბმისი ექსპერიმენტი (Walmart_Sales_Data_Exploration), რომელშიც შესაბამის run-ში (EDA_Walmart_Sales) დალოგილია artifacts-ში ყველა plot-ი, მათ შორის:
- ACF & PACF გრაფიკები
- სეზონური დეკომპოზიცია
- მოძრავი საშუალო და სტანდარტული გადახრა
- სხვაობის სერიები (Differencing)

---
# model_experiment_XGBoost.ipynb/model_experiment_LightGBM.ipynb ნოუთბუქების აღწერა:

## ლოგირება:
Mlflow-ზე არის ცალ-ცალკე ექპერიმენტებად (XGBoost_Training და LightGBM_Training), ხოლო თითოეულში არის შესაბამისი run-ები:
- Feature_Engineering
- EDA
- Feature_Selection
- RandomSearchCV_XGBoost / RandomSearchCV_LightGBM
- Final_Pipeline_XGBRegressor / Final_Pipeline_LGBMRegressor


##  1. Feature_Engineering 

### გამოყოფილია თარიღის ახალი სვეტები:
- `Year`, `Month`, `Day`, `WeekOfYear` (`Date` სვეტიდან)

### გამოყოფილია არდადეგების ახალი სვეტები:
- `SuperbowlWeek`, `LaborDay`, `Thanksgiving`, `Christmas`
- `IsHoliday`-ის ბინარული კოდირება: {True: 1, False: 0}

### მაღაზიის ტიპის კოდირება:
- `StoreType` encoding: {A': 3, 'B': 2, 'C': 1}

## 2. EDA

მონაცემთა შესწავლა გამოიყენება ტენდენციების, სეზონურობისა და სხვა დაკვირვებების გამოსავლენად.

### ვიზუალიზაციები (დალოგილია artifacts-ში):
სხვადასხვა Boxplot-ები, Barplot-ები, heatmap-ები, target ხვლადისა და სხვა ცვლადების დამოკიდებულებები:

- Average Monthly Sales Per Store
- Average Weekly Sales Per Department
- Average Weekly Sales Per Department Yearly
- Average Weekly Sales Per Store
- Average Weekly Sales Per Store Yearly
- Average Weekly Sales Per Year
- Correlation Matrix
- Missing Values Plot
- Percent of Store Types
- CPI VS Sales
- Fuel Price VS Sales
- Holidays VS NonHolidays
- Store Size VS Sales
- Temperature VS Sales
- Unemployment VS Sales
- Week VS Sales


## 3. Feature_Selection 

- target-თან სუსტად კორელირებული ცვლადების ამოღება
- Markdown სვეტების ამოღება ბევრი გამოტოვებული მნიშვნელობის გამო (Markdown-ებითაც ვცადე, გამოტოვებულების 0-ებით შევსებით, მაგრამ ნაკლები სქორი ჰქონდა ვალიდაციაზე)
- მულტიკოლინარობის თავიდან აცილება (მაგ., `Month` და `Day` არ გამოიყენება, დატოვებულია `WeekOfYear` მათ ნაცვლად)


##  4. Preprocessing Pipeline
- გაერთიანებულია Feature Selection და Feature Engineering (fit, transform ფუნქციები) მიდგომები და შეფუთულია კლასად


## 5. მონაცემების გაყოფა

- **დროის მიხედვით გაყოფა** სასწავლო და ვალიდაციის მონაცემებად (cutoff თარიღის მიხედვით, დაახლ. ბოლო 20% არის აღებული ვალიდაციისთვის)


## 6. ქროს ვალიდაცია (RandomSearchCV)

- გამოყენებულია `PredefinedSplit` სატრენინგო მონაცემებზე და custom WMAE scorer-ი შეფასებისთვის
- გამოყენებულია ჰიპერპარამეტრების ტუნინგი და ამორჩეულია 50 კომბინაცია დიდი ჰიპერპარამეტრების სივრციდან (საუკეთესო შედეგის მქონე მოდელის პარამეტრები და სქორი დალოგილია შესაბამის run-ში)


## 7. საუკეთესო მოდელის ტრენინგი (Final_Pipeline)

- საუკეთესო შედეგის მქონე მოდელისთვის შექმნილია Final Pipeline, რომელიც დატრენინგებულია სატრენინგო მონაცემებზე და ვალიდაცია ხდება სავალიდაციო მონაცემებზე
- საუკეთესო მოდელის ჰიპერპარამეტრები, სხვადასხვა მეტრიკები (wmae, mae, rmse, r2) როგორც სატრენინგო, ასევე სავალიდაციო მონაცემებზე დალოგილია შესაბამის run-ში
- მოდელი დალოგილია და შენახულია model registry-ში, საიდანაც დალოუდდება და ხდება სატესტო მონაცემებზე პროგნოზი

## შედეგები

- Final XGBoost Model: WMAE_val = ~1551, Kaggle Private Score = ~3141, Kaggle Public Score = ~3055
- Final LightGBM Model: WMAE_val = ~2044, Kaggle Private Score = ~3778, Kaggle Public Score = ~3738 

---

# model_experiment_ARIMA_SARIMA_SARIMAX.ipynb ნოუთბუქის აღწერა:

## გამოყენებულია სამი მოდელი: 
- ARIMA (Autoregressive Integrated Moving Average): ძირითადი დროის სერიების მოდელი, რომელიც არ ითვალისწინებს სეზონურობას.
- SARIMA (Seasonal Autoregressive Integrated Moving Average): ARIMA-ს გაფართოება, რომელიც მოიცავს სეზონურ კომპონენტებს.
- SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors): SARIMA-ს შემდგომი გაფართოება, რომელიც აერთიანებს დამატებით გარე (ეგზოგენურ) ცვლადებს პროგნოზირების გასაუმჯობესებლად.

## WalmartSalesPreprocessingPipeline კლასი:
- აერთიანებს ფაილებს
- ავსებს გამოტოვებულ მნიშვნელობებს
- ქმნის დროზე დაფუძნებულ და ციკლურ მახასიათებლებს (Year, Month, WeekOfYear, DayOfWeek, sin_week, cos_week, sin_month, cos_month, sin_dayofweek, cos_dayofweek)
- ახდენს მაღაზიის ტიპის კოდირებას (one-hot encoding)
- ქმნის ახალ Holiday ცვლადებს (SuperBowl, LaborDay, Thanksgiving, Christmas)

## analyze_stationarity ფუნქცია: 
- დროის სერიების სტაციონარობის ანალიზი ADF და KPSS ტესტების გამოყენებით, ACF/PACF პლოტებთან ერთად

## StoreDeptForecaster კლასი:
- იყენებს pmdarima.auto_arima-ს ფუნქციას, რათა ავტომატურად მოძებნოს ოპტიმალური ARIMA/SARIMA/SARIMAX პარამეტრები (p, d, q და სეზონური P, D, Q, s) კონკრეტული მაღაზია-დეპარტამენტის დროის სერიისთვის
- ხდება მოდელის მორგება სასწავლო მონაცემებზე და გაყიდვების პროგნოზირებას ვალიდაციის ან სატესტო პერიოდისთვის, ეგზოგენური ცვლადების გათვალისწინებით SARIMAX მოდელების შემთხვევაში

## ტრენინგი და მოდელის შერჩევა:
- გამოიყენება joblib.Parallel მრავალი მაღაზია-დეპარტამენტის კომბინაციის პროგნოზირების პროცესის დასაჩქარებლად
- ვალიდაციის მონაცემებზე მიღებული WMAE-ის საფუძველზე ხდება საუკეთესო მოდელის იდენტიფიცირება
- საუკეთესო მოდელები რეგისტრირდება MLflow-ში
- თითოეული მოდელის შესაბამის ექსპერიმენტში დალოგილია შესაბამისი run-ები, პარამეტრები და მეტრიკები

## ლოგირება:
Mlflow-ზე არის ცალ-ცალკე ექპერიმენტებად (ARIMA_Training, SARIMA_Training, SARIMAX_Training), ხოლო თითოეულში არის შესაბამისი run-ები:
- Preprocessing_ARIMA / Preprocessing_SARIMA / Preprocessing_SARIMAX
- CV_ARIMA / CV_SARIMA / CV_SARIMAX
- Best_Model_ARIMA / Best_Model_SARIMA / Best_Model_SARIMAX

## შედეგები:
- val_WMAE ~ 2165 (ARIMA), ~ 1811 (SARIMA/SARIMAX)
- მხოლოდ 20 store-dept წყვილია აღებული (დიდი დრო მიჰქონდა ტრენინგს): unique_series_ids_for_cv = unique_series_ids_for_cv[:20], ამიტომ submission-ზე ისედაც ცუდი შედეგი ექნება
---

---
# model_experiment_DLinear.ipynb ნოუთბუქის აღწერა:

## გამოყენებულია DLinear მოდელი:
- DLinear (Decomposition Linear Model): ღრმა სწავლების დროის სერიების მოდელი, რომელიც იყენებს ციკლურ კომპონენტთა დეკომპოზიციას და ხაზოვან პროგნოზირებას.
- ეფექტურად იყენებს წარსული გაყიდვების და ექსოგენური მახასიათებლების კომბინაციას სეზონური და ტრენდული კომპონენტების პროგნოზირებისთვის.

## WalmartDLinearPreprocessor კლასი:
- აერთიანებს გაყიდვების, დამატებითი მახასიათებლების და მაღაზიების მეტამონაცემების ფაილებს.
- ავსებს გამოტოვებულ მნიშვნელობებს (Markdown-ებში ნულებით, სხვა მახასიათებლებში ინტერპოლაციით).
- იყენებს რობუსტურ სკეილინგს (მედიანა და IQR) მაღაზია-დეპარტამენტის სერიების მიხედვით.
- ასახელებს სვეტებს NeuralForecast სტანდარტებით (`series_id` → `unique_id`, `Date` → `ds`, `Weekly_Sales` → `y`).
- უზრუნველყოფს თანმიმდევრულ წინამუშავებას `fit`, `transform`, `fit_transform` მეთოდებით.

## DLinearModelTrainer კლასი (ან ტრენინგის პროცესი):
- მოდელის კონფიგურაცია და ტრენინგი NeuralForecast DLinear მოდელით.
- ტრენინგი ხდება კვირის კალენდარულ სიხშირეზე `W-FRI` (კვირის ბოლო პარასკევი).
- ძირითადი პარამეტრები: პროგნოზის ჰორიზონტი 39 კვირა, შესვლის სიგრძე 52 კვირა, რობუსტური სკეილინგი, ადრეული გაჩერება, ბეჩ საიზი და სხვა.
- ტრენინგი იყენებს ისტორიულ გაყიდვებს და ექსოგენურ მახასიათებლებს.

## ვალიდაცია და შეფასება:
- ვალიდაცია ცალკე სპლიტით ბოლო 13 კვირაზე.
- პროგნოზები ინვერსულად გადამუშავებულია რეალურ ერთეულებამდე.
- შეფასებულია MAE და wMAE რეალურ მნიშვნელობებზე.
- შედეგები ლოგირდებიან Weights & Biases (WandB) სისტემაში.

## ექსპერიმენტების მართვა და ლოგირება:
- ტრენინგი და ვალიდაცია სხვადასხვა ექსპერიმენტებში, თითოეული ექსპერიმენტი რეგისტრირებულია WandB-ზე.
- ჩანაწერილია პარამეტრები, შეფასების მეტრიკები და პროგნოზების ვიზუალიზაციები.
- შესაძლებელი ჰიპერპარამეტრების ოპტიმიზაცია და მოდელის გაუმჯობესება.

---

# model_experiment_TFT.ipynb ნოუთბუქის აღწერა

## მოდელის მიმოხილვა
- **Temporal Fusion Transformer (TFT)** — თანამედროვე დროის სერიული მონაცემების პროგნოზირების DNN არქიტექტურა, რომელიც ეფექტურად იყენებს სტატიკურ და დროის მიხედვით ცვალებად მახასიათებლებს. მოდელი შეიცავს ყურადღების მექანიზმებს და ინტერპრეტაციულ ინსტრუმენტებს პროგნოზების ახსნისთვის.

## WalmartTFTPreprocessor კლასი
- მონაცემების გაერთიანება `train_df`, `features_df` და `stores_df`-დან.
- გამოტოვებული Markdown მახასიათებლების შევსება ნულებით და შესაბამისი ინფორმაციის შენახვა (`_was_missing`).
- დროზე დაფუძნებული მახასიათებლების გენერირება: წელი, თვე, კვირის დღე, კვირა, თვის დასაწყისი.
- ციკლური მახასიათებლების (თვე, კვირა, კვირის დღე) სინუს/კოსინუსის ენკოდირება.
- სტატიკური კატეგორიული მახასიათებლების (Store, Dept, Type) ლეიბლ ენკოდირება.
- უნიკალური სერიის იდენტიფიკატორის შექმნა `series_id` ფორმატში.
- გარდამუშავებული DataFrame და გამოყენებული მახასიათებლების სია.

## ტრენინგი და შეფასება
- ტრენინგისთვის გამოყენებულია `pytorch-forecasting`-ის TimeSeriesDataSet და TemporalFusionTransformer.
- მონაცემები დაყოფილია ტრენინგსა და ვალიდაციად.
- ტრენინგი ხორციელდება PyTorch Lightning გარემოში, გამოყენებულია EarlyStopping და LearningRateMonitor კალბეკები.
- ძირითადი პარამეტრები:
  - `learning_rate=0.03`
  - `hidden_size=16`
  - `attention_head_size=4`
  - `dropout=0.1`
- ტრენინგი შეზღუდულია 30 ეპოქით (კოდის ნაწილი შეიცავს `limit_train_batches` და `limit_val_batches` პარამეტრებს ტესტირების სიჩქარისთვის).
- ტრენინგის პროცესი ლოგირდება `wandb` პლატფორმაზე.

## შედეგები
- მიღებულია პროგნოზები ვალიდაციაზე.
- შეფასდა MAE და წონიანი MAE (WMAE), განსაკუთრებით ყურადღების გამახვილებით არდადეგებსა და ჩვეულებრივ დღეებზე.
- TFT მოდელმა აჩვენა მაღალი სიზუსტე და სტაბილურობა ბაზრის სხვა მოდელებთან შედარებით.

## ლოგირება და ექსპერიმენტების მართვა
- ექსპერიმენტები რეგისტრირებულია `wandb`-ში, სადაც მონიტორინგდება სწავლების და ვალიდაციის პროცესი.
- შესაძლებელი გახდა ტრენინგის პარამეტრების ოპტიმიზაცია და მოდელის გაანალიზება ლოგების საფუძველზე.

---


