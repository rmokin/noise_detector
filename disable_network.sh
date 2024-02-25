#!/bin/bash
CURRENT_DIR=$(dirname "$0")
CURRENT_SCRIPT_NAME=$(basename "$0")
EVENT_FILE="$CURRENT_DIR/disable_network.event"
NETWORK_INTERFACE="wlp2s0"
RULE="POSTROUTING -o $NETWORK_INTERFACE -j MASQUERADE -t nat"
IPTABLE_ENABLING_RULE="-A $RULE" 
IPTABLE_DISABLING_RULE="-D $RULE"
CHECKING_RULE_IS_ENABLED=""

LOG_FILE="/var/log/noisedetector/$CURRENT_SCRIPT_NAME.log"

function check(){
	echo "$(iptables-save | grep -E "^-A\s+POSTROUTING.*$NETWORK_INTERFACE" --)"
}

function showMessage() {
	DATE=$(date +"%Y-%m-%d %T")
	STR="$DATE $1"
	echo $STR
	echo $STR >> $LOG_FILE
}

if test -f $EVENT_FILE; 
then
	showMessage "$EVENT_FILE is found"
	showMessage "Start disabling network rule: $RULE"
	CHECKING_RULE_IS_ENABLED="$(check)"
	showMessage "Check rule first state:$CHECKING_RULE_IS_ENABLED"
	if [ -z "$CHECKING_RULE_IS_ENABLED" ]
	then
		showMessage "Network rule: ' $RULE ' isn't enabled"
	else
		showMessage "Disabling network rule: $IPTABLE_DISABLING_RULE"
		iptables $IPTABLE_DISABLING_RULE
		CHECKING_RULE_IS_ENABLED="$(check)"
		showMessage "Check rule last state:$CHECKING_RULE_IS_ENABLED"
	fi
	
else
	showMessage "$EVENT_FILE isn't found"
	showMessage "Start enabling network rule: $RULE"
	CHECKING_RULE_IS_ENABLED="$(check)"
	showMessage "Check rule first state:$CHECKING_RULE_IS_ENABLED"
	if [ -z "$CHECKING_RULE_IS_ENABLED" ]
	then
		showMessage "Enabling network rule: $IPTABLE_ENABLING_RULE"
		iptables $IPTABLE_ENABLING_RULE
		CHECKING_RULE_IS_ENABLED="$(check)"
		showMessage "Check rule last state:$CHECKING_RULE_IS_ENABLED"
	else
		showMessage "Network rule: ' $RULE ' is already enabled"
	fi
fi
