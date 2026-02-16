# mu_alpha ML 모델 스펙 (state/mu_alpha_model.pt)

이 프로젝트의 런타임 ML 추론은 `main_engine_mc_v2_final.py::_predict_mu_ml()`에서 수행됩니다.
모델 파일은 기본적으로 `state/mu_alpha_model.pt` 경로에 둡니다.

## 입력 스펙
- 입력 데이터: 최근 특징 시퀀스 (기본 3채널)
  - `feature[0]`: 로그수익률 `log(close_t / close_{t-1})`
  - `feature[1]`: 로그 거래량 변화 `log(vol_t / vol_{t-1})` (볼륨 없으면 0)
  - `feature[2]`: 절대 수익률 `abs(feature[0])`
- 시퀀스 길이: 모델의 `seq_len` 속성 또는 `ML_SEQ_LEN` 값 사용
- 입력 텐서 shape: `(batch=1, seq_len, n_features)`
  - `n_features`는 모델 속성 `n_features`에 따름 (기본 3, 구버전은 1)
- dtype: `float32`
- 정규화: **시퀀스 내부 z-score (feature별 mean/std)** 적용

예시 (추론 입력 생성):
```python
rets = np.diff(np.log(np.maximum(closes, 1e-12))).astype(np.float32)
vol_rets = np.diff(np.log(np.maximum(volumes, 1e-12))).astype(np.float32)
abs_rets = np.abs(rets).astype(np.float32)
feats = np.stack([rets, vol_rets, abs_rets], axis=1)
feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-6)
seq = torch.tensor(feats[-seq_len:], dtype=torch.float32).view(1, -1, feats.shape[1])
```

## 출력 스펙
- 출력: **단일 스칼라** (또는 (1,1) 형태)
- 의미: **per-bar log-return 기대값 (mu_bar)**
- 런타임에서 annualize 적용:
  - `mu_ann = mu_bar * (31536000 / bar_seconds)`
  - 현재 bar_seconds=60으로 가정

## 모델 저장 형식
- **TorchScript** (`torch.jit.save`) 또는 일반 PyTorch `torch.save` 모두 지원
- 로딩 우선순위:
  1) `torch.jit.load(path)`
  2) 실패 시 `torch.load(path)`

## 권장 출력 스케일
- 모델 출력은 **작은 값**이어야 합니다.
- 예: `mu_bar`가 1e-4 ~ 1e-3 범위면 연율 변환 시 현실적인 값이 됩니다.

## 방향성 확률 모델 사용 시
- 만약 모델이 상승 확률 `p`를 출력한다면,
  - `mu_bar = log(p/(1-p)) * scale` 형태로 변환 후 출력하도록 모델에 내장하거나,
  - 추론 직후 변환 로직을 추가하세요.

## 주의사항
- 모델이 출력하는 값은 반드시 **per-bar 로그수익률** 기준이어야 합니다.
- 너무 큰 출력은 mu_alpha cap에 의해 잘리며, 실제 성능이 악화됩니다.
- 구버전(1채널 입력) 모델도 동작하도록 런타임에서 자동 감지합니다.
