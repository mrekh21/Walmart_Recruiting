# Kaggle Competition
https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/overview 

# Mlflow Link
https://dagshub.com/mrekh21/Walmart_Recruiting.mlflow/#/experiments/0

# Dagshub Link
https://dagshub.com/mrekh21/Walmart_Recruiting


# Walmart-ის გაყიდვების პროგნოზირება

ეს პროექტი მიზნად ისახავს Walmart-ის მაღაზიების **ყოველკვირეული გაყიდვების პროგნოზირებას**. გამოყენებულია **Tree-Based Models** (LightGBM, XGBoost), **Classical Statistical Time-Series Models** (ARIMA, SARIMA, SARIMAX), **Deep Learning**-ის მოდელები, მონაცემთა ანალიზი (EDA) და სხვადასხვა time-series/seasonal plot-ები. ექსპერიმენტები დალოგილია **MLflow**-ით და **DagsHub**-ით.


# მონაცემები

პროექტში გამოყენებული მონაცემები შედგება შემდეგი ფაილებისგან:

- `train.csv.zip` – ისტორიული გაყიდვების მონაცემები
- `test.csv.zip` – სატესტო მონაცემები პროგნოზისთვის
- `features.csv.zip` – ეკონომიკური და მარკეტინგული მახასიათებლები
- `stores.csv` – მაღაზიების ტიპისა და ზომის მონაცემები



# დირექტორიის სტრუქტურა
├── Walmart_Recruiting_Data_Exploration.ipynb
├── model_experiment_XGBoost.ipynb
├── model_experiment_LightGBM.ipynb
├── model_experiment_ARIMA_SARIMA_SARIMAX.ipynb
├── README.md



# Walmart_Recruiting_Data_Exploration.ipynb ნოუთბუქის აღწერა:
- თითოეული csv ფაილის feature-ების ტიპები, გამოტოვებული მონაცემების რაოდენობა და ა.შ არის შესწავლილი
- სატრენინგო მონაცემების სხვადასხვა time-series/seasonal plot-ებია წარმოდგენილი matplotlib, seaborn, plotly, statsmodels ბიბლიოთეკების გამოყენებით
- mlflow-ზე დალოგილია შესაბმისი ექსპერიმენტი (Walmart_Sales_Data_Exploration), რომელშიც შესაბამის run-ში (EDA_Walmart_Sales) დალოგილია artifacts-ში ყველა plot-ი, მათ შორის:
- ACF & PACF გრაფიკები
- სეზონური დეკომპოზიცია
- მოძრავი საშუალო და სტანდარტული გადახრა
- სხვაობის სერიები (Differencing)


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
- Markdown სვეტების ამოღება ბევრი გამოტოვებული მნიშვნელობის გამო
- მულტიკოლინარობის თავიდან აცილება (მაგ., `Month` და `Day` არ გამოიყენება, დატოვებულია `WeekOfYear` მათ ნაცვლად)


##  4. Preprocessing Pipeline-ში გაერთიანებულია Feature Selection და Feature Engineering (fit, transform ფუნქციები)


## 5. მონაცემების გაყოფა

- **დროის მიხედვით გაყოფა** სასწავლო და ვალიდაციის მონაცემებად (cutoff თარიღის მიხედვით, დაახლ. ბოლო 20% არის აღებული ვალიდაციისთვის)


## 6. ქროს ვალიდაცია (RandomSearchCV)

- გამოყენებულია `PredefinedSplit` და custom WMAE scorer-ი 
- გამოყენებულია ჰიპერპარამეტრების ტუნინგი და ამორჩეულია 50 კომბინაცია დიდი ჰიპერპარამეტრების სივრციდან (საუკეთესო შედეგის მქონე მოდელის პარამეტრები და სქორი დალოგილია შესაბამის run-ში)


## 7. საუკეთესო მოდელის ტრენინგი (Final_Pipeline)

- საუკეთესო შედეგის მქონე მოდელისთვის შექმნილია Final Pipeline, რომელიც დატრენინგებულია სატრენინგო მონაცემებზე და ვალიდაცია ხდება სავალიდაციო მონაცემებზე
- საუკეთესო მოდელის ჰიპერპარამეტრები, სხვადასხვა მეტრიკები (wmae, mae, rmse, r2) როგორც სატრენინგო, ასევე სავალიდაციო მონაცემებზე დალოგილია შესაბამის run-ში
- მოდელი დალოგილია და შენახულია model registry-ში, საიდანაც დალოუდდება და ხდება სატესტო მონაცემებზე პროგნოზი

---





