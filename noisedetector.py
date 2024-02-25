#!/usr/bin/env python3
import numpy as np
import sounddevice as sd
import soundfile as sf
import argparse
import sys
import re
import os
import subprocess
import json
import logging
import queue
import time
from importlib import reload
import signal

class SignalHandler:
    shutdown_requested = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.request_shutdown)
        signal.signal(signal.SIGTERM, self.request_shutdown)

    def request_shutdown(self, *args):
        global logger
        logger.info("Request to shutdown received, stopping")
        send_message_to_telegram(message="Bye-Bye NoiseDetector, see you later!!!")
        self.shutdown_requested = True

    def can_run(self):
        return not self.shutdown_requested



#To use beeper:https://wiki.archlinux.org/title/PC_speaker (3.2 Run as non-root user)
#https://forums.linuxmint.com/viewtopic.php?t=368886 
#You can mute the microphone with
#amixer set Capture nocap
#and unmute the microphone with
#amixer set Capture cap
#and set volume
#amixer set Capture 5%hisdt
#Enable Internetsharing:  iptables -A POSTROUTING -o wlp2s0 -j MASQUERADE -t nat
#Disaple Internetsharing:  iptables  -D POSTROUTING -o wlp2s0 -j MASQUERADE -t nat
#wlp2s0 - Internet network interface should be specifed
#View rules:  sudo iptables -t nat -L

def current_milli_time():
    return round(time.time() * 1000)
    
def rms_flat(a):  # from matplotlib.mlab
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return np.sqrt(np.mean(np.absolute(a)**2))

def move (y, x):
    print("\033[%d;%dH" % (y, x))

def audio_callback_test(indata, frames, time, status):
    volume_norm = np.linalg.norm(indata) * 10
    print ("|" * (int(volume_norm) % 239))
 
def listen(audio_callback, duration=None, deviceID=None):
    def __callback__(indata, frames, time, status):
        global QUEUE 
        QUEUE.put({
            "indata": indata.copy(),
            "frames": frames,
            "time": time,
            "status": status,
        })
    
    def __work__():
        global QUEUE, CURRENT_COUNT_SIGNALS, SETTINGS
        while not QUEUE.empty():
            q = QUEUE.get()
            CURRENT_COUNT_SIGNALS += 1
            if CURRENT_COUNT_SIGNALS > int(SETTINGS['listening']['ignore_first_signals']):
                audio_callback(**q)
            """
            CURRENT_COUNT_SIGNALS, SETTINGS
            CURRENT_COUNT_SIGNALS += 1
            if CURRENT_COUNT_SIGNALS > int(SETTINGS['listening']['ignore_first_signals']):
                audio_callback(indata, frames, time, status)
            """
        
    with sd.InputStream(
        callback=__callback__, 
        device=deviceID,
        clip_off=True,
        dither_off=True,
        prime_output_buffers_using_stream_callback=True
    ):
        global QUEUE, SIGNAL_HANDLER
        
        if duration is None:
            while SIGNAL_HANDLER.can_run():
                __work__()
        else:
            sd.sleep(duration * 1000)
            __work__()

def beep(frequency=2000, lenght=100, repeat=1, delay_between_repeats=100, callback_success=None, callback_error=None):
    run_subprocess(
        [f'beep -f {frequency} -l {lenght} -r {repeat} -d {delay_between_repeats}'], 
        callback_success,
        callback_error,
        shell=True, 
    )

def send_message_to_telegram(message="test", callback_success=None, callback_error=None):
    run_subprocess(
        [f'./send_message_to_telegram.sh "{message}"'], 
        callback_success,
        callback_error,
        shell=True, 
    )

def run_subprocess(cmnd, callback_success=None, callback_error=None, shell=False):
    try:
        output = subprocess.check_output(cmnd, stderr=subprocess.STDOUT, shell=shell)
    except subprocess.CalledProcessError as exc:
        if callable(callback_error):
            callback_error(exc.returncode, exc.output)
        return None
    else:
        if callable(callback_success):
            callback_success(output)
        return output

def set_sensetivity(sensetivity, device='Capture', callback_success=None, callback_error=None):
    #amixer set Capture 5%
    run_subprocess(
        ['amixer','set', device, f'{sensetivity}%'], 
        callback_success,
        callback_error,
    )

def get_sensetivity(device='Capture', callback_success=None, callback_error=None):
    #amixer get Capture
    return run_subprocess(
        ['amixer','get', device], 
        callback_success,
        callback_error,
    )

def get_volume(device='Master', callback_success=None, callback_error=None):
    # amixer sget 'Master'
    return run_subprocess(
        ['amixer','sget', device], 
        callback_success,
        callback_error,
    )

def set_volume(volume, device='Master', callback_success=None, callback_error=None):
    # amixer sget 'Master' 100%
    run_subprocess(
        ['amixer','sset', device, f'{volume}%'], 
        callback_success,
        callback_error,
    )
    
    
def mute(device, callback_success=None, callback_error=None):
    #amixer set Capture nocap
    run_subprocess(
        ['amixer','set', args.device,'nocap'], 
        callback_success,
        callback_error,
    )
 
def unmute(device, callback_success=None, callback_error=None):
    #amixer set Capture cap
    run_subprocess(
        ['amixer','set', args.device,'cap'], 
        callback_success,
        callback_error,
    )
    
def parseArgs():
    parser = argparse.ArgumentParser(add_help=True)
    
    parser.add_argument( 
        '--list-devices', 
        action='store_true',
        help='''Show list of audio devices and Exit'''
    )
    parser.add_argument( 
        '--start-lisent', 
        action='store_true',
        help='''Start script as service'''
    )
    parser.add_argument(
        '--test-microphone', 
        action='store_true',
        help='''Show out realtime audio volume as bar. 
        During test press Ctrl+C for Exit.'''
    )
    parser.add_argument(
        '--test-beeper', 
        type=int,
        metavar='REPEAT',
        default=0,
        help='''Test sound of beeper.'''
    )
    parser.add_argument(
        '--test-telegram', 
        type=str,
        metavar='TELEGRAM_MESSAGE',
        default=0,
        help='''Test send message to telegram.'''
    )
    parser.add_argument(
        '--get-sensetivity', 
        action='store_true',
        help='''Show current microphone sensetivity via "amixer" tools'''
    )
    parser.add_argument(
        '--set-sensetivity',
        type=int,
        metavar='SENSETIVITY',
        help='''Set microphone sensetivity in persent via "amixer" tools'''
    )
    parser.add_argument(
        '--get-volume', 
        action='store_true',
        help='''Show current volume via "amixer" tools'''
    )
    parser.add_argument(
        '--set-volume',
        type=int,
        metavar='VOLUME',
        help='''Set volume in persent via "amixer" tools'''
    )
    parser.add_argument(
        '--mute',
        action='store_true',
        help='''Mute microphone via "amixer" tools'''
    )
    parser.add_argument(
        '--unmute',
        action='store_true',
        help='''Unmute microphone via "amixer" tools'''
    )
    parser.add_argument(
        '--play-sound', 
        metavar='PATH_TO_WAV',
        help='''Play WAV file.'''
    )
    

    args = parser.parse_args()
    
    return args, parser

def settings(path='noisedetector.config.json'):
    global logger, SETTINGS
    loaded_settings = {}
    try:
        if os.path.isfile(path):
            with open(path) as f:
                loaded_settings = json.load(f)
        else:
            logger.debug(f"Can't find config. Use defaults")
    except Exception as e:
        logger.critical(f"Can't load config. Error: {str(e)}")
        sys.exit(1)
    else:
        SETTINGS.update(loaded_settings)
        SETTINGS["log"]["level"] = int(SETTINGS["log"]["level"])
        if SETTINGS["log"]["level"] < 10:
            SETTINGS["log"]["level"] = logging.NOTSET
        elif 10 <= SETTINGS["log"]["level"] < 20:
            SETTINGS["log"]["level"] = logging.DEBUG
        elif 20 <= SETTINGS["log"]["level"] < 30:
            SETTINGS["log"]["level"] = logging.INFO
        elif 30 <= SETTINGS["log"]["level"] < 40:
            SETTINGS["log"]["level"] = logging.WARNING
        elif 40 <= SETTINGS["log"]["level"] < 50:
            SETTINGS["log"]["level"] = logging.ERROR
        else:
            SETTINGS["log"]["level"] = logging.CRITICAL
        
        logger.setLevel(SETTINGS["log"]["level"])
        log_formatter = logging.Formatter(SETTINGS["log"]["format"])

        log_file_handler = logging.FileHandler(f"/var/log/noisedetector/{os.path.basename(__file__)}.log", mode='a')   
        log_file_handler.setFormatter(log_formatter)
        logger.addHandler(log_file_handler)

        log_console_handler = logging.StreamHandler()
        log_console_handler.setFormatter(log_formatter)
        logger.addHandler(log_console_handler)
        

def load_state(path='noisedetector.state.json'):
    global logger, STATE
    loaded_state = {}
    try:
        if os.path.isfile(path):
            with open(path) as f:
                loaded_state = json.load(f)
        else:
            logger.debug(f"Can't find state. Use init values")
    except Exception as e:
        logger.warning(f"Can't load state. Error: {str(e)}")
    else:
        STATE.update(loaded_state)
        
def save_state(path='noisedetector.state.json'):
    global logger, STATE
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(STATE, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Can't save state. Error: {str(e)}")
    else:
        pass


def play_sound(path=None, needWait=True):
    if path:
        logger.debug(f'Play sound: {path}')
        (data, fs) = sf.read(path, dtype='float32')
        sd.play(data,fs)
        if needWait:
            return sd.wait()       
        
def terminate_network():
    logger.info(f'Write disable_network.event')
    send_message_to_telegram(message="Disable network")
    f = open("disable_network.event", "w")
    f.write("Netwotk need disabling!")
    f.close()
    
def unterminate_network():
    logger.info(f'Delete disable_network.event')
    send_message_to_telegram(message="Enable network")
    if os.path.exists("disable_network.event"):
        os.remove("disable_network.event")


def change_stress_level(increase=0):
    global STATE, ASS_IS_ON_FIRE
    logger.info(f"Current stress level:{STATE['stress_level']}")
    logger.info(f"Increase:{increase}")
    
    is_changed = False
    
    last_stress_level = float(STATE['stress_level'])
    new_stress_level = float(STATE['stress_level']) + increase
    if new_stress_level < 0:
        new_stress_level = 0
    elif new_stress_level > 10:
        new_stress_level = 10
    
    if last_stress_level != new_stress_level or ASS_IS_ON_FIRE == 0:
        is_changed = True
       
    
    if is_changed:
        STATE['stress_level'] = new_stress_level
        save_state()
        str_info=f"New stress level:{STATE['stress_level']} (max 10)"
        logger.info(str_info)
        send_message_to_telegram(message=str_info)
        
        if increase > 0:
            if 0 < new_stress_level <= 3:
                run_subprocess(
                    [f'./noisedetector.py --test-beep 1'], 
                    shell=True, 
                )
            elif 3 < new_stress_level <= 5:
                run_subprocess(
                    [f'./noisedetector.py --test-beep 3'], 
                    shell=True, 
                )
            elif 5 < new_stress_level <= 7:
                run_subprocess(
                    [f'./noisedetector.py --test-beep 5'], 
                    shell=True, 
                )
            elif 7 < new_stress_level <= 9:
                run_subprocess(
                    [f'./noisedetector.py --play-sound stop.wav'], 
                    shell=True, 
                )
            else: 
                if not ASS_IS_ON_FIRE:
                    run_subprocess(
                        [f'./noisedetector.py --play-sound terminate.wav'], 
                        shell=True, 
                    )
                ASS_IS_ON_FIRE = 1
                terminate_network()
            
        else:
            if STATE['stress_level'] < 9: 
                ASS_IS_ON_FIRE = 0
                unterminate_network()
        
def angry_up():
    global ANGRY_UP_TIME, CALM_DOWN_TIME, SETTINGS
    t = current_milli_time() - ANGRY_UP_TIME
    logger.debug(f"Try angry up {t} > {float(SETTINGS['listening']['angry_up_delay'])}")
    trick_detect(0)
    if current_milli_time() - ANGRY_UP_TIME > float(SETTINGS['listening']['angry_up_delay']):
        change_stress_level(1)
        CALM_DOWN_TIME= ANGRY_UP_TIME = current_milli_time()

def trick_detect(increase=1):
    global SETTINGS, STATE, ASS_IS_ON_FIRE
    
    if increase > 0:
        str_info=f"Trick was detected! Current trick count={float(STATE['tricks_count'])} (max: {float(SETTINGS['listening']['tricks_count_max'])})"
        logger.info(str_info)
        run_subprocess(
            [f'./noisedetector.py --test-beep 2'], 
            shell=True, 
        )
        if float(STATE['tricks_count']) >= float(SETTINGS['listening']['tricks_count_max']):
            if ASS_IS_ON_FIRE == 0:
                send_message_to_telegram(message=str_info)
                change_stress_level(10)
        else:
            STATE['tricks_count'] += increase
            save_state()
    else:
        if float(STATE['tricks_count']) > 0:
            send_message_to_telegram(message="Thanks for don't use a trick")
        STATE['tricks_count'] = 0
        save_state()
        
    
    
def calm_down():
    global CALM_DOWN_TIME, ANGRY_UP_TIME, SETTINGS
    t = current_milli_time() - CALM_DOWN_TIME
    logger.debug(f"Try calm down {t} > {float(SETTINGS['listening']['calm_down_delay'])}")
    trick_detect(0)
    if current_milli_time() - CALM_DOWN_TIME > float(SETTINGS['listening']['calm_down_delay']):
        change_stress_level(-1)
        ANGRY_UP_TIME = CALM_DOWN_TIME = current_milli_time()

def main():
    
    global SETTINGS, STATE
    
    args, parser = parseArgs()
    
    if args.test_microphone:
        print('Start a microphone test. For Exit press Ctrl+C.')
        listen(audio_callback=audio_callback_test)
        sys.exit(0)
        
    if args.test_beeper:
        r = int(args.test_beeper)
        beep(repeat=1 if r <= 0 else r)
        sys.exit(0)
    
    if args.test_telegram:
        message = args.test_telegram
        print(f'Start a telegram send message test:{message}')
        def on_error(code, stderr):
            print(f"Can't send message to telegram: {stderr.decode('utf-8')}")
        def on_success(stdout):
            print(stdout.decode('utf-8'))
        send_message_to_telegram(
            message=message,
            callback_success=on_success,
            callback_error=on_error
        )
        sys.exit(0)
    
    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)
        
    if args.get_sensetivity:
        def on_error(code, stderr):
            print(f"Can't get sensetivity: {stderr.decode('utf-8')}")
        def on_success(stdout):
            match = re.search(r"\[(.+?)%\]\s+?\[(.+?)dB\]", stdout.decode('utf-8'))
            if match:
                persent, decibels = match.group(1), match.group(2)
                print(f'Sensetivity: {persent}% [{decibels}dB]')
            else:
                print(f"Can't parse reuslt of amixer tools. Result: {stdout.decode('utf-8')}")
        get_sensetivity(
            callback_success=on_success,
            callback_error=on_error
        )
        sys.exit(0)
    
    if args.set_sensetivity:
        def on_error(code, stderr):
            print(f"Can't set sensetivity: {stderr.decode('utf-8')}")
        def on_success(stdout):
            print(stdout.decode('utf-8'))
        set_sensetivity(
            sensetivity=args.set_sensetivity,
            callback_success=on_success,
            callback_error=on_error
        )
        sys.exit(0)
        
    if args.get_volume:
        def on_error(code, stderr):
            print(f"Can't get volume: {stderr.decode('utf-8')}")
        def on_success(stdout):
            match = re.search(r"\[(.+?)%\]\s+?\[(.+?)dB\]", stdout.decode('utf-8'))
            if match:
                persent, decibels = match.group(1), match.group(2)
                print(f'Volume: {persent}% [{decibels}dB]')
            else:
                print(f"Can't parse reuslt of amixer tools. Result: {stdout.decode('utf-8')}")
        get_volume(
            callback_success=on_success,
            callback_error=on_error
        )
        sys.exit(0)
    
    if args.set_volume:
        def on_error(code, stderr):
            print(f"Can't set sensetivity: {stderr.decode('utf-8')}")
        def on_success(stdout):
            print(stdout.decode('utf-8'))
        set_volume(
            volume=args.set_volume,
            callback_success=on_success,
            callback_error=on_error
        )
        sys.exit(0)
    
    if args.play_sound:
        play_sound(args.play_sound, True)
    
    if args.mute:
        def on_error(code, stderr):
            print(f"Can't mute: {stderr.decode('utf-8')}")
        def on_success(stdout):
            print(stdout.decode('utf-8'))
        mute(
            device=args.device,
            callback_success=on_success,
            callback_error=on_error
        )
        sys.exit(0)
    
    if args.unmute:
        def on_error(code, stderr):
            print(f"Can't unmute: {stderr.decode('utf-8')}")
        def on_success(stdout):
            print(stdout.decode('utf-8'))
        unmute(
            device=args.device,
            callback_success=on_success,
            callback_error=on_error
        )
        sys.exit(0)
    
    if args.start_lisent:
        logger.info(f"Start noise analizator.")
        
        load_state('noisedetector.state.json')
        change_stress_level()
        
        send_message_to_telegram(message=f"Hi! I'm NoiseDetector. I'll control your noise and your tricks. Have a Good Day!!! My current stress level is {STATE['stress_level']}")
        
        logger.debug(f"Allowed noise volume: {SETTINGS['listening']['allowed_noise_volume_max']}")
        logger.debug(f"AVG signal delay: {SETTINGS['listening']['avg_signal_delay']}")
        
        logger.debug(f"Set volume {SETTINGS['play']['volume']}")
        
        def on_error_volume(code, stderr):
            logger.error(f"Can't set volume: {stderr.decode('utf-8')}")
        
        def on_success_volume(stdout):
            logger.debug(f"Check volume to...")
            
            def on_error(code, stderr):
                logger.error(f"Can't get volume: {stderr.decode('utf-8')}")
            
            def on_success(stdout):
                match = re.search(r"\[(.+?)%\]\s+?\[(.+?)dB\]", stdout.decode('utf-8'))
                if match:
                    persent, decibels = float(match.group(1)) / 100., float(match.group(2)) 
                    logger.debug(f"Volume is: {persent}[{decibels}dB]")
                else:
                    logger.error(f"Can't parse reuslt of amixer tools. Result: {stdout.decode('utf-8')}")    

            get_volume(
                callback_success=on_success,
                callback_error=on_error
            )
            
        set_volume(
            volume=int(SETTINGS['play']['volume']),
            callback_success=on_success_volume,
            callback_error=on_error_volume
        )
        
        logger.debug(f"Set microphone sensetivity to: {SETTINGS['listening']['microphone_sensetivity']}")
        
        def on_error_sensetivity(code, stderr):
            logger.error(f"Can't set microphone sensetivity: {stderr.decode('utf-8')}")
            
        def on_success_sensetivity(stdout):
            logger.debug(f"Check microphone sensetivity to...")
            def on_error(code, stderr):
                logger.error(f"Can't get sensetivity: {stderr.decode('utf-8')}")
            def on_success(stdout):
                match = re.search(r"\[(.+?)%\]\s+?\[(.+?)dB\]", stdout.decode('utf-8'))
                if match:
                    persent, decibels = float(match.group(1)) / 100., float(match.group(2)) 
                    logger.debug(f"Microphone sensetivity is: {persent}[{decibels}dB]")
                    def audio_callback_work(indata, frames, time, status):
                        nonlocal decibels
                        global SETTINGS, AVG_NOISE_VOLUME, LAST_SIGNAL_TIME, AVG_SIGNAL_TIME, AVG_SIGNAL_DELAY
                        LAST_SIGNAL_TIME = current_milli_time()
                        volume_norm = np.linalg.norm(indata) * 10
                        volume_db = float(20 * np.log10(rms_flat(indata) / 2e-5))
                        
                        AVG_NOISE_VOLUME = volume_norm if AVG_NOISE_VOLUME < 0 else (AVG_NOISE_VOLUME + volume_norm) / 2.0
                        if current_milli_time() - AVG_SIGNAL_TIME >= float(SETTINGS['listening']['avg_signal_delay']):
                            logger.debug(f"Average noise volume: {AVG_NOISE_VOLUME:.2f}")
                            logger.debug(f"Allowed noise volume: {SETTINGS['listening']['allowed_noise_volume_max']:.2f}")
                            AVG_SIGNAL_TIME = current_milli_time()
                            if AVG_NOISE_VOLUME <= float(SETTINGS['listening']['allowed_noise_volume_min']):
                                logger.info(f"TricksDetected")
                                trick_detect()
                            elif float(SETTINGS['listening']['allowed_noise_volume_min']) <= AVG_NOISE_VOLUME <= float(SETTINGS['listening']['allowed_noise_volume_max']):
                                logger.info(f"CalmDown")
                                calm_down()
                            else:
                                logger.info(f"AngryUp")
                                angry_up()
                            
                                
                           
                    listen(audio_callback=audio_callback_work)
                else:
                    logger.error(f"Can't parse reuslt of amixer tools. Result: {stdout.decode('utf-8')}")
                    
            get_sensetivity(
                callback_success=on_success,
                callback_error=on_error
            )
        
       
        set_sensetivity(
            sensetivity=int(SETTINGS['listening']['microphone_sensetivity']),
            callback_success=on_success_sensetivity,
            callback_error=on_error_sensetivity
        )



DEFAULT_SETTINGS={
    "log":{
        "level": logging.INFO,
        "format": "%(name)s %(asctime)s %(levelname)s %(message)s",
    },
    "play":{
		"volume": 95
	},
    "listening":{
        "ignore_first_signals": 64,
        "microphone_sensetivity": 50,
        "allowed_noise_volume_max": 60,
        "allowed_noise_volume_min": 20,
        "tricks_count_max": 10,
        "avg_signal_delay":10000, 
        "angry_up_delay": 60000,
        "calm_down_delay": 120000,
    }
}
STATE = {
    'ass_is_on_fire': 0,
    'stress_level': 0,
    'tricks_count': 0,
}
SETTINGS = {**DEFAULT_SETTINGS}
CURRENT_COUNT_SIGNALS = 0
ANGRY_UP_TIME = current_milli_time()
CALM_DOWN_TIME = current_milli_time()
LAST_SIGNAL_TIME = current_milli_time()
AVG_SIGNAL_TIME = current_milli_time()
AVG_NOISE_VOLUME = -1
ASS_IS_ON_FIRE = 0
QUEUE = queue.Queue()
SIGNAL_HANDLER = SignalHandler()



logger = logging.getLogger(__file__)


if __name__ == "__main__":
    settings('noisedetector.config.json')
    main()
