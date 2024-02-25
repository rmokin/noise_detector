#!/bin/bash
CHAT_ID="-1002034368686"
TELEGRAM_TOKEN="6981171926:AAFtKSkftKcao_1yw2oPxxghGFTpzkE5mWM"
CURRENT_SCRIPT_NAME=$(basename "$0")

LOG_FILE="/var/log/noisedetector/$CURRENT_SCRIPT_NAME.log"

function showMessage() {
	DATE=$(date +"%Y-%m-%d %T")
	STR="$DATE $1"
	echo $STR
	echo $STR >> $LOG_FILE
}

DATA="{\"chat_id\":\"$CHAT_ID\",\"text\":\"$1\"}"
URL="https://api.telegram.org/bot$TELEGRAM_TOKEN/sendMessage"

showMessage "Send to: $URL"
showMessage "Data: $DATA"

RESPONSE=$(curl -X POST -H "Content-Type: application/json" -d "$DATA" $URL)
showMessage "Response: $RESPONSE"


