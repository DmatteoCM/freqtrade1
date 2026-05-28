# Freqtrade — Config & Strategies

Branch `freqShow` — sincronizzazione tra macchina locale e server **showbox** (`192.168.50.200`).

## Struttura

```
compose/
├── docker-compose.yaml
├── user_data/
│   ├── strategies/          # file .py e .json delle strategie
│   ├── config_dcaProgressive.json   # bot attivo (8 coppie USDC)
│   ├── config_adaptiveSpotX.json    # strategia in test
│   └── config_secrets.json          # ⚠ NON nel repo — solo sul server
```

## Strategia attiva

**DcaProgressive** — DCA con tranche progressive (6→33 USDC), trailing stoploss condizionale, time_exit adattivo.

- Coppie: BTC ETH SOL XRP ADA DOGE APT DOT (tutte /USDC)
- Timeframe: 15m + 1h informativo
- Wallet dry run: 1600 USDC

## Setup iniziale su showbox

```bash
# 1. Aggiungere la chiave SSH di showbox su GitHub (una tantum)
cat ~/.ssh/id_ed25519.pub
# → copiare su github.com → Settings → SSH Keys

# 2. Aggiungere github.com agli known_hosts
ssh-keyscan github.com >> ~/.ssh/known_hosts

# 3. Clonare il repo nella cartella freqtrade
git clone -b freqShow git@github.com:DmatteoCM/freqtrade1.git /home/showbox/compose
```

## Sincronizzare showbox (aggiornamento)

```bash
cd /home/showbox/compose
git pull origin freqShow
```

## Riavviare il bot dopo un aggiornamento

```bash
cd /home/showbox/compose
docker compose up -d --force-recreate freqtrade
```

## Workflow tipico

```bash
# 1. Modifica i file in locale (o dalla versione web di Claude)
# 2. Commit e push
git add user_data/strategies/DcaProgressive.py
git commit -m "descrizione modifica"
git push origin freqShow

# 3. Su showbox: pull e riavvio
ssh showbox@192.168.50.200
cd /home/showbox/compose && git pull origin freqShow
docker compose up -d --force-recreate freqtrade
```

## Backtest in locale

```bash
cd /home/matteo/compose

# DcaProgressive
docker run --rm -v "./user_data:/freqtrade/user_data" freqtradeorg/freqtrade:stable \
  backtesting \
  --config /freqtrade/user_data/config_dcaProgressive.json \
  --strategy DcaProgressive \
  --timerange 20260101- --breakdown month

# AdaptiveSpotX
docker run --rm -v "./user_data:/freqtrade/user_data" freqtradeorg/freqtrade:stable \
  backtesting \
  --config /freqtrade/user_data/config_adaptiveSpotX.json \
  --strategy AdaptiveSpotX \
  --timerange 20260101- --breakdown month
```

## Download dati

```bash
cd /home/matteo/compose

# DcaProgressive (15m + 1h)
docker run --rm -v "./user_data:/freqtrade/user_data" freqtradeorg/freqtrade:stable \
  download-data \
  --config /freqtrade/user_data/config_dcaProgressive.json \
  --timerange 20251101- --timeframes 15m 1h

# AdaptiveSpotX (5m + 1h + 4h)
docker run --rm -v "./user_data:/freqtrade/user_data" freqtradeorg/freqtrade:stable \
  download-data \
  --config /freqtrade/user_data/config_adaptiveSpotX.json \
  --timerange 20251101- --timeframes 5m 1h 4h
```

## Note importanti

- `config_secrets.json` contiene API keys Binance, token Telegram e credenziali FreqUI — **non viene mai committato**
- Su showbox è già presente e non va sovrascritto durante il pull
- Il docker-compose su showbox usa la porta `65000` invece di `8080` — modificare prima di fare il deploy se si clona da zero
