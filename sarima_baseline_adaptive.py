"""
自适应SARIMA Baseline - Walk-Forward Forecasting
================================================
根据每个region的平稳性自适应选择差分参数d

改进点：
1. 对每个region进行ADF检验
2. 非平稳序列使用d=1，平稳序列使用d=0
3. 强制平稳性约束（enforce_stationarity=True）
4. 数值稳定性检查
5. 记录每个region的参数选择

作者：Claude Code
日期：2026-03-10
"""

import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*100)
print("自适应SARIMA Baseline - Walk-Forward Forecasting")
print("="*100)

# ============================================================================
# 步骤1: 数据加载和检查
# ============================================================================
print("\n[1/8] 数据加载和检查...")

df = pd.read_csv(r'C:\Users\mxx\Desktop\清洗数据\delivery_region_hour_clean.csv')
print(f"数据shape: {df.shape}")
print(f"列名: {list(df.columns)}")

# 转换时间
df['date'] = pd.to_datetime(df['date'])
df['datetime'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')
df = df.sort_values(['region_id', 'datetime']).reset_index(drop=True)

print(f"Region数: {df['region_id'].nunique()}")
print(f"时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
print(f"总记录数: {len(df)}")

# 添加核心时段标记
df['is_core_hour'] = df['hour'].between(9, 19).astype(int)

# ============================================================================
# 步骤2: 数据划分
# ============================================================================
print("\n[2/8] 数据划分...")

train_start = pd.Timestamp('2017-05-01')
train_end = pd.Timestamp('2017-08-08 23:00:00')
val_start = pd.Timestamp('2017-08-09')
val_end = pd.Timestamp('2017-09-05 23:00:00')
test_start = pd.Timestamp('2017-09-06')
test_end = pd.Timestamp('2017-10-31 23:00:00')

train_df = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
val_df = df[(df['datetime'] >= val_start) & (df['datetime'] <= val_end)].copy()
test_df = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

print(f"训练集: {train_start.date()} 至 {train_end.date()}")
print(f"  记录数: {len(train_df)}, 天数: {train_df['date'].nunique()}")
print(f"验证集: {val_start.date()} 至 {val_end.date()}")
print(f"  记录数: {len(val_df)}, 天数: {val_df['date'].nunique()}")
print(f"测试集: {test_start.date()} 至 {test_end.date()}")
print(f"  记录数: {len(test_df)}, 天数: {test_df['date'].nunique()}")

# ============================================================================
# 步骤3: 对每个region进行平稳性检验
# ============================================================================
print("\n[3/8] 对每个region进行ADF平稳性检验...")

regions = sorted(df['region_id'].unique())
region_stationarity = {}

print(f"检验{len(regions)}个regions的平稳性...")

for region in tqdm(regions, desc="ADF检验"):
    region_train = train_df[train_df['region_id'] == region].sort_values('datetime')

    if len(region_train) < 100:
        # 数据不足，默认为非平稳
        region_stationarity[region] = {
            'is_stationary': False,
            'adf_statistic': None,
            'p_value': None,
            'reason': 'insufficient_data'
        }
        continue

    try:
        # ADF检验
        result = adfuller(region_train['count'].values, autolag='AIC')
        adf_stat = result[0]
        p_value = result[1]
        is_stationary = p_value < 0.05  # p<0.05认为是平稳的

        region_stationarity[region] = {
            'is_stationary': is_stationary,
            'adf_statistic': adf_stat,
            'p_value': p_value,
            'reason': 'adf_test'
        }
    except:
        # ADF检验失败，默认为非平稳
        region_stationarity[region] = {
            'is_stationary': False,
            'adf_statistic': None,
            'p_value': None,
            'reason': 'adf_failed'
        }

# 统计平稳性
stationary_count = sum(1 for r in region_stationarity.values() if r['is_stationary'])
non_stationary_count = len(regions) - stationary_count

print(f"\n平稳性检验结果:")
print(f"  平稳序列: {stationary_count} ({stationary_count/len(regions)*100:.1f}%)")
print(f"  非平稳序列: {non_stationary_count} ({non_stationary_count/len(regions)*100:.1f}%)")

# 显示非平稳的regions
non_stationary_regions = [r for r, info in region_stationarity.items() if not info['is_stationary']]
print(f"\n非平稳regions: {non_stationary_regions}")

# ============================================================================
# 步骤4: 参数选择（使用验证集）
# ============================================================================
print("\n[4/8] 参数选择（使用验证集）...")

# 选择一个代表性region进行参数搜索
representative_region = df.groupby('region_id')['count'].sum().idxmax()
print(f"使用Region {representative_region} 进行参数搜索")

region_train = train_df[train_df['region_id'] == representative_region].sort_values('datetime')
region_val = val_df[val_df['region_id'] == representative_region].sort_values('datetime')

train_series = region_train['count'].values
val_series = region_val['count'].values

# 检查该region的平稳性
is_stationary = region_stationarity[representative_region]['is_stationary']
print(f"Region {representative_region} 平稳性: {'平稳' if is_stationary else '非平稳'}")

# 根据平稳性选择候选参数
if is_stationary:
    # 平稳序列：d=0
    param_candidates = [
        (1, 0, 1, 1, 0, 1, 24),  # SARIMA(1,0,1)(1,0,1)[24]
        (2, 0, 1, 1, 0, 1, 24),  # SARIMA(2,0,1)(1,0,1)[24]
        (1, 0, 2, 1, 0, 1, 24),  # SARIMA(1,0,2)(1,0,1)[24]
    ]
else:
    # 非平稳序列：d=1
    param_candidates = [
        (1, 1, 1, 1, 0, 1, 24),  # SARIMA(1,1,1)(1,0,1)[24]
        (2, 1, 1, 1, 0, 1, 24),  # SARIMA(2,1,1)(1,0,1)[24]
        (1, 1, 2, 1, 0, 1, 24),  # SARIMA(1,1,2)(1,0,1)[24]
    ]

best_val_mae = np.inf
best_params = None

print("搜索参数空间...")
for p, d, q, P, D, Q, s in param_candidates:
    try:
        model = SARIMAX(train_series,
                       order=(p, d, q),
                       seasonal_order=(P, D, Q, s),
                       enforce_stationarity=True,   # 强制平稳性
                       enforce_invertibility=True)  # 强制可逆性
        fitted = model.fit(disp=False, maxiter=50, method='lbfgs')

        # 在验证集上评估
        forecast = fitted.forecast(steps=len(val_series))
        val_mae = np.mean(np.abs(val_series - forecast))

        print(f"  SARIMA({p},{d},{q})({P},{D},{Q})[{s}]: Val MAE = {val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_params = (p, d, q, P, D, Q, s)
    except:
        print(f"  SARIMA({p},{d},{q})({P},{D},{Q})[{s}]: 拟合失败")

if best_params is None:
    print("警告: 所有参数都失败，使用默认参数")
    best_params = (1, 1, 1, 1, 0, 1, 24) if not is_stationary else (1, 0, 1, 1, 0, 1, 24)

p_best, d_best, q_best, P_best, D_best, Q_best, s_best = best_params
print(f"\n最优参数: SARIMA({p_best},{d_best},{q_best})({P_best},{D_best},{Q_best})[{s_best}]")
print(f"验证集MAE: {best_val_mae:.4f}")

# ============================================================================
# 步骤5: 为每个region确定参数
# ============================================================================
print("\n[5/8] 为每个region确定自适应参数...")

region_params = {}

for region in regions:
    is_stationary = region_stationarity[region]['is_stationary']

    if is_stationary:
        # 平稳序列：使用d=0的最优参数
        region_params[region] = {
            'order': (p_best, 0, q_best),
            'seasonal_order': (P_best, D_best, Q_best, s_best),
            'reason': 'stationary'
        }
    else:
        # 非平稳序列：使用d=1的最优参数
        region_params[region] = {
            'order': (p_best, 1, q_best),
            'seasonal_order': (P_best, D_best, Q_best, s_best),
            'reason': 'non_stationary'
        }

print(f"参数配置完成:")
print(f"  使用d=0的regions: {sum(1 for p in region_params.values() if p['order'][1] == 0)}")
print(f"  使用d=1的regions: {sum(1 for p in region_params.values() if p['order'][1] == 1)}")

# ============================================================================
# 步骤6: 严格的Walk-Forward预测
# ============================================================================
print("\n[6/8] 严格的Walk-Forward预测...")
print("策略: 每天滚动一次（每天00:00作为forecast origin）")

test_dates = sorted(test_df['date'].unique())

print(f"Region数: {len(regions)}")
print(f"测试天数: {len(test_dates)}")
print(f"总预测次数: {len(regions) * len(test_dates)}")

# 存储所有预测
all_predictions = []
fallback_info = []
region_performance = {}

print("\n开始Walk-Forward预测...")

for region in tqdm(regions, desc="Region进度"):
    # 获取该region的完整数据
    region_data = df[df['region_id'] == region].sort_values('datetime').reset_index(drop=True)

    # 获取该region的参数
    params = region_params[region]
    order = params['order']
    seasonal_order = params['seasonal_order']

    # 检查数据质量
    train_region = region_data[region_data['datetime'] <= train_end]
    if len(train_region) < 100 or train_region['count'].std() < 0.1:
        fallback_info.append({
            'region_id': region,
            'reason': 'insufficient_data_or_low_variance',
            'train_samples': len(train_region),
            'std': train_region['count'].std()
        })

        # 使用训练集小时平均作为fallback
        hour_avg = train_region.groupby('hour')['count'].mean().to_dict()

        for test_date in test_dates:
            for h in range(24):
                target_time = pd.Timestamp(test_date) + pd.Timedelta(hours=h)
                true_row = region_data[region_data['datetime'] == target_time]

                if len(true_row) > 0:
                    all_predictions.append({
                        'model': 'fallback_ha',
                        'region_id': region,
                        'forecast_origin': pd.Timestamp(test_date),
                        'target_time': target_time,
                        'horizon': h + 1,
                        'y_true': true_row['count'].values[0],
                        'y_pred': hour_avg.get(h, 0),
                        'params': 'fallback'
                    })
        continue

    # 对每个测试日进行预测
    for test_date in test_dates:
        forecast_origin = pd.Timestamp(test_date)

        # 只使用forecast origin之前的数据
        history = region_data[region_data['datetime'] < forecast_origin]

        if len(history) < 100:
            continue

        try:
            # 训练模型（使用所有可用历史数据）
            history_series = history['count'].values

            model = SARIMAX(
                history_series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=True,   # 强制平稳性
                enforce_invertibility=True   # 强制可逆性
            )

            fitted_model = model.fit(disp=False, maxiter=50, method='lbfgs')

            # 预测未来24小时
            forecast = fitted_model.forecast(steps=24)

            # 数值稳定性检查
            max_forecast = np.max(np.abs(forecast))
            if max_forecast > 1000 or np.isnan(max_forecast) or np.isinf(max_forecast):
                raise ValueError(f"Numerical instability: max_forecast={max_forecast}")

            # 保存预测结果
            for h in range(24):
                target_time = forecast_origin + pd.Timedelta(hours=h)
                true_row = region_data[region_data['datetime'] == target_time]

                if len(true_row) > 0:
                    pred_value = forecast.iloc[h] if hasattr(forecast, 'iloc') else forecast[h]
                    all_predictions.append({
                        'model': 'sarima_adaptive',
                        'region_id': region,
                        'forecast_origin': forecast_origin,
                        'target_time': target_time,
                        'horizon': h + 1,
                        'y_true': true_row['count'].values[0],
                        'y_pred': max(0, pred_value),
                        'params': f"({order[0]},{order[1]},{order[2]})"
                    })

        except Exception as e:
            # 如果预测失败，使用训练集平均
            hour_avg = train_region.groupby('hour')['count'].mean().to_dict()

            for h in range(24):
                target_time = forecast_origin + pd.Timedelta(hours=h)
                true_row = region_data[region_data['datetime'] == target_time]

                if len(true_row) > 0:
                    all_predictions.append({
                        'model': 'fallback_ha',
                        'region_id': region,
                        'forecast_origin': forecast_origin,
                        'target_time': target_time,
                        'horizon': h + 1,
                        'y_true': true_row['count'].values[0],
                        'y_pred': hour_avg.get(h, 0),
                        'params': 'fallback'
                    })

print(f"\n预测完成，共生成 {len(all_predictions)} 条预测")
print(f"使用fallback的region数: {len(fallback_info)}")

# 转换为DataFrame
predictions_df = pd.DataFrame(all_predictions)

# ============================================================================
# 步骤7: 计算评估指标
# ============================================================================
print("\n[7/8] 计算评估指标...")

# 添加小时信息
predictions_df['hour'] = predictions_df['target_time'].dt.hour
predictions_df['is_core_hour'] = predictions_df['hour'].between(9, 19).astype(int)

# 计算误差
predictions_df['ae'] = np.abs(predictions_df['y_true'] - predictions_df['y_pred'])
predictions_df['se'] = (predictions_df['y_true'] - predictions_df['y_pred']) ** 2

# 整体指标
overall_mae = predictions_df['ae'].mean()
overall_rmse = np.sqrt(predictions_df['se'].mean())
overall_mape = (predictions_df['ae'] / (predictions_df['y_true'] + 1e-8)).mean() * 100

# 核心时段 vs 非核心时段
core_df = predictions_df[predictions_df['is_core_hour'] == 1]
noncore_df = predictions_df[predictions_df['is_core_hour'] == 0]

core_mae = core_df['ae'].mean()
core_rmse = np.sqrt(core_df['se'].mean())
noncore_mae = noncore_df['ae'].mean()
noncore_rmse = np.sqrt(noncore_df['se'].mean())

# 业务加权MAE
business_weighted_mae = 0.9266 * core_mae + 0.0734 * noncore_mae

print(f"\n整体指标:")
print(f"  MAE:  {overall_mae:.4f}")
print(f"  RMSE: {overall_rmse:.4f}")
print(f"  MAPE: {overall_mape:.2f}%")

print(f"\n核心时段 (9-19点):")
print(f"  MAE:  {core_mae:.4f}")
print(f"  RMSE: {core_rmse:.4f}")
print(f"  样本数: {len(core_df)}")

print(f"\n非核心时段:")
print(f"  MAE:  {noncore_mae:.4f}")
print(f"  RMSE: {noncore_rmse:.4f}")
print(f"  样本数: {len(noncore_df)}")

print(f"\n业务加权MAE: {business_weighted_mae:.4f}")

# Region-wise分析
region_mae = predictions_df.groupby('region_id')['ae'].mean().reset_index()
region_mae.columns = ['region_id', 'mae']
region_mae = region_mae.sort_values('mae')

print(f"\nRegion-wise MAE统计:")
print(f"  最小MAE: {region_mae['mae'].min():.4f} (Region {region_mae.iloc[0]['region_id']})")
print(f"  最大MAE: {region_mae['mae'].max():.4f} (Region {region_mae.iloc[-1]['region_id']})")
print(f"  中位MAE: {region_mae['mae'].median():.4f}")

# 模型使用统计
model_counts = predictions_df.groupby('model').size()
print(f"\n模型使用统计:")
for model_name, count in model_counts.items():
    pct = count / len(predictions_df) * 100
    print(f"  {model_name}: {count} ({pct:.1f}%)")

# ============================================================================
# 步骤8: 保存结果
# ============================================================================
print("\n[8/8] 保存结果...")

# 保存指标
metrics = {
    'model': 'SARIMA_Adaptive',
    'parameters': 'Adaptive (d=0 for stationary, d=1 for non-stationary)',
    'Overall_MAE': overall_mae,
    'Overall_RMSE': overall_rmse,
    'Overall_MAPE': overall_mape,
    'Core_MAE': core_mae,
    'Core_RMSE': core_rmse,
    'NonCore_MAE': noncore_mae,
    'NonCore_RMSE': noncore_rmse,
    'BusinessWeighted_MAE': business_weighted_mae,
    'Total_Predictions': len(predictions_df),
    'Core_Predictions': len(core_df),
    'NonCore_Predictions': len(noncore_df),
    'Fallback_Regions': len(fallback_info),
    'Stationary_Regions': stationary_count,
    'NonStationary_Regions': non_stationary_count,
    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('sarima_adaptive_baseline_metrics.csv', index=False, encoding='utf-8-sig')
print("[OK] 指标已保存: sarima_adaptive_baseline_metrics.csv")

# 保存详细预测结果（采样保存）
sample_predictions = predictions_df.sample(min(10000, len(predictions_df)), random_state=42)
sample_predictions.to_csv('sarima_adaptive_predictions_sample.csv', index=False, encoding='utf-8-sig')
print("[OK] 预测样本已保存: sarima_adaptive_predictions_sample.csv")

# 保存region-wise分析
region_mae.to_csv('sarima_adaptive_region_mae.csv', index=False, encoding='utf-8-sig')
print("[OK] Region分析已保存")

# 保存平稳性检验结果
stationarity_df = pd.DataFrame([
    {
        'region_id': r,
        'is_stationary': info['is_stationary'],
        'adf_statistic': info['adf_statistic'],
        'p_value': info['p_value'],
        'reason': info['reason'],
        'd_parameter': 0 if info['is_stationary'] else 1
    }
    for r, info in region_stationarity.items()
])
stationarity_df.to_csv('sarima_adaptive_stationarity.csv', index=False, encoding='utf-8-sig')
print("[OK] 平稳性检验结果已保存: sarima_adaptive_stationarity.csv")

# 保存fallback信息
if fallback_info:
    fallback_df = pd.DataFrame(fallback_info)
    fallback_df.to_csv('sarima_adaptive_fallback_info.csv', index=False, encoding='utf-8-sig')
    print(f"[OK] Fallback信息已保存: {len(fallback_info)} regions")

print("\n" + "="*100)
print("自适应SARIMA Baseline完成!")
print("="*100)
print(f"\n关键结果:")
print(f"  整体MAE: {overall_mae:.4f}")
print(f"  业务加权MAE: {business_weighted_mae:.4f}")
print(f"  核心时段MAE: {core_mae:.4f}")
print(f"  非核心时段MAE: {noncore_mae:.4f}")
print(f"\n平稳性统计:")
print(f"  平稳regions (d=0): {stationary_count}")
print(f"  非平稳regions (d=1): {non_stationary_count}")
print(f"\n所有结果文件已保存到当前目录")
