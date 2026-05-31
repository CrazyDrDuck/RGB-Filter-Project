#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.1),
    on March 15, 2026, at 18:31
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.1'
expName = 'Train1'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = (1920,1200)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\lille\\NewOneDrive\\OneDrive\\Emilie\\MasterThesisProject\\Train1.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('error')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Welcome" ---
    Welcome_info = visual.TextStim(win=win, name='Welcome_info',
        text='Welcome to the experiment.\n\nYou will be tasked with finding the right color in an image based on the RGB representations as seen through 4 filters or using the values in percentages.\nYou can switch filters using spacebar as many times as you like, and there is no time constraint.\n\nThe color buttons are randomly ordered, but this order will not change.\n\nThis session does not have an end point.\nRather, you get to press a button whenever you wish to stop due to fatigue at the end of each trial.',
        font='Arial',
        units='pix', pos=(0, 0), draggable=False, height=42.0, wrapWidth=1500.0, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    PressEnter = keyboard.Keyboard(deviceName='defaultKeyboard')
    PressEnterToContinue = visual.TextStim(win=win, name='PressEnterToContinue',
        text='Press "Enter" to continue.',
        font='Arial',
        units='pix', pos=(0, -400), draggable=False, height=50.0, wrapWidth=900.0, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    # Set experiment start values for variable component BreakIt
    BreakIt = ''
    BreakItContainer = []
    # Set experiment start values for variable component trial_counter
    trial_counter = 0
    trial_counterContainer = []
    # Run 'Begin Experiment' code from BreakItInit
    BreakIt == False
    trial_counter = 0
    colorList = []
    
    # --- Initialize components for Routine "initiate" ---
    # Set experiment start values for variable component img_Var
    img_Var = ''
    img_VarContainer = []
    # Set experiment start values for variable component logical_Var
    logical_Var = ''
    logical_VarContainer = []
    # Set experiment start values for variable component corrColor_Var
    corrColor_Var = ''
    corrColor_VarContainer = []
    # Set experiment start values for variable component corrR_Var
    corrR_Var = ''
    corrR_VarContainer = []
    # Set experiment start values for variable component corrG_Var
    corrG_Var = ''
    corrG_VarContainer = []
    # Set experiment start values for variable component corrB_Var
    corrB_Var = ''
    corrB_VarContainer = []
    # Set experiment start values for variable component filterCount
    filterCount = 0
    filterCountContainer = []
    # Set experiment start values for variable component img_Var1
    img_Var1 = ''
    img_Var1Container = []
    
    # --- Initialize components for Routine "trial_1" ---
    img = visual.ImageStim(
        win=win,
        name='img', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=1.0, pos=[0,0], draggable=False, size=1.0,
        color='white', colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    spacebar = keyboard.Keyboard(deviceName='defaultKeyboard')
    blue = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,10),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='blue',
        depth=-3
    )
    blue.buttonClock = core.Clock()
    yellow = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,-50),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='yellow',
        depth=-4
    )
    yellow.buttonClock = core.Clock()
    purple = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,-110),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='purple',
        depth=-5
    )
    purple.buttonClock = core.Clock()
    cyan = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,-170),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='cyan',
        depth=-6
    )
    cyan.buttonClock = core.Clock()
    red = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,-290),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='red',
        depth=-7
    )
    red.buttonClock = core.Clock()
    green = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,-230),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='green',
        depth=-8
    )
    green.buttonClock = core.Clock()
    Instructions = visual.TextStim(win=win, name='Instructions',
        text='',
        font='Arial',
        units='pix', pos=(0, 500), draggable=False, height=36.0, wrapWidth=1400.0, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    # Set experiment start values for variable component answer
    answer = ""
    answerContainer = []
    RBGInfo = visual.TextStim(win=win, name='RBGInfo',
        text='RGB values: 0-100\n\nR:        G:        B:',
        font='Arial',
        units='pix', pos=(300, 300), draggable=False, height=42.0, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-11.0);
    R = visual.TextStim(win=win, name='R',
        text='',
        font='Arial',
        units='pix', pos=(165, 220), draggable=False, height=42.0, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-12.0);
    G = visual.TextStim(win=win, name='G',
        text='',
        font='Arial',
        units='pix', pos=(300, 220), draggable=False, height=42.0, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-13.0);
    B = visual.TextStim(win=win, name='B',
        text='',
        font='Arial',
        units='pix', pos=(435, 220), draggable=False, height=42.0, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-14.0);
    
    # --- Initialize components for Routine "Pause350" ---
    text_8 = visual.TextStim(win=win, name='text_8',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "confidence" ---
    slider = visual.Slider(win=win, name='slider',
        startValue=1, size=(1000, 100), pos=(-50, 0), units='pix',
        labels=("Not at all","A little", "Somewhat", "Fairly", "Very \n confident"), ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=[], opacity=1.0,
        labelColor=(-1.0000, -1.0000, -1.0000), markerColor=('#000000'), lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=36.0,
        flip=False, ori=0.0, depth=0, readOnly=False)
    text_9 = visual.TextStim(win=win, name='text_9',
        text='How confident were you in your choice?',
        font='Arial',
        pos=(0, 0.25), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mouse_2 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_2.mouseClock = core.Clock()
    button2 = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(0,-400),units='pix',
        letterHeight=30.0,
        size=(300,100), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='button2',
        depth=-3
    )
    button2.buttonClock = core.Clock()
    text_answer = visual.TextStim(win=win, name='text_answer',
        text='',
        font='Arial',
        pos=(0, 0.35), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    blue_3 = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,10),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='blue_3',
        depth=-5
    )
    blue_3.buttonClock = core.Clock()
    yellow_3 = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,-50),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='yellow_3',
        depth=-6
    )
    yellow_3.buttonClock = core.Clock()
    purple_3 = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,-110),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='purple_3',
        depth=-7
    )
    purple_3.buttonClock = core.Clock()
    cyan_3 = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,-170),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='cyan_3',
        depth=-8
    )
    cyan_3.buttonClock = core.Clock()
    red_3 = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,-290),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='red_3',
        depth=-9
    )
    red_3.buttonClock = core.Clock()
    green_3 = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,-230),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='green_3',
        depth=-10
    )
    green_3.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "Pause350" ---
    text_8 = visual.TextStim(win=win, name='text_8',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "confirm" ---
    Button3 = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(0,-400),units='pix',
        letterHeight=30.0,
        size=(300,100), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='Button3',
        depth=0
    )
    Button3.buttonClock = core.Clock()
    correctAnswer = visual.TextStim(win=win, name='correctAnswer',
        text='',
        font='Arial',
        units='pix', pos=(0, -150), draggable=False, height=56.0, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    wrongAnswer = visual.TextStim(win=win, name='wrongAnswer',
        text='',
        font='Arial',
        units='pix', pos=(0, -150), draggable=False, height=56.0, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-2.0);
    corrColor_2 = visual.TextStim(win=win, name='corrColor_2',
        text='',
        font='Arial',
        units='pix', pos=(0, -150), draggable=False, height=56.0, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-3.0);
    blue_2 = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,10),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='blue_2',
        depth=-4
    )
    blue_2.buttonClock = core.Clock()
    yellow_2 = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,-50),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='yellow_2',
        depth=-5
    )
    yellow_2.buttonClock = core.Clock()
    purple_2 = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,-110),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='purple_2',
        depth=-6
    )
    purple_2.buttonClock = core.Clock()
    cyan_2 = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,-170),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='cyan_2',
        depth=-7
    )
    cyan_2.buttonClock = core.Clock()
    red_2 = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,-290),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='red_2',
        depth=-8
    )
    red_2.buttonClock = core.Clock()
    green_2 = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(750,-230),units='pix',
        letterHeight=30.0,
        size=(250,50), 
        ori=0.0
        ,borderWidth=20.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='bottom-center',
        name='green_2',
        depth=-9
    )
    green_2.buttonClock = core.Clock()
    RBGInfo_2 = visual.TextStim(win=win, name='RBGInfo_2',
        text='RGB values: 0-100\n\nR:        G:        B:',
        font='Arial',
        units='pix', pos=(300, 300), draggable=False, height=42.0, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-10.0);
    R_2 = visual.TextStim(win=win, name='R_2',
        text='',
        font='Arial',
        units='pix', pos=(165, 220), draggable=False, height=42.0, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-11.0);
    G_2 = visual.TextStim(win=win, name='G_2',
        text='',
        font='Arial',
        units='pix', pos=(300, 220), draggable=False, height=42.0, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-12.0);
    B_2 = visual.TextStim(win=win, name='B_2',
        text='',
        font='Arial',
        units='pix', pos=(435, 220), draggable=False, height=42.0, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-13.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    BaselineOverrideInfo = visual.TextStim(win=win, name='BaselineOverrideInfo',
        text='',
        font='Arial',
        units='pix', pos=(-550, 400), draggable=False, height=50.0, wrapWidth=650.0, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-16.0);
    
    # --- Initialize components for Routine "PauseEnd" ---
    text_11 = visual.TextStim(win=win, name='text_11',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "congrats" ---
    GoodJob = visual.TextStim(win=win, name='GoodJob',
        text='',
        font='Arial',
        units='pix', pos=(0, 0), draggable=False, height=50.0, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Welcome" ---
    # create an object to store info about Routine Welcome
    Welcome = data.Routine(
        name='Welcome',
        components=[Welcome_info, PressEnter, PressEnterToContinue],
    )
    Welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for PressEnter
    PressEnter.keys = []
    PressEnter.rt = []
    _PressEnter_allKeys = []
    # store start times for Welcome
    Welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Welcome.tStart = globalClock.getTime(format='float')
    Welcome.status = STARTED
    thisExp.addData('Welcome.started', Welcome.tStart)
    Welcome.maxDuration = None
    # keep track of which components have finished
    WelcomeComponents = Welcome.components
    for thisComponent in Welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Welcome" ---
    thisExp.currentRoutine = Welcome
    Welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Welcome_info* updates
        
        # if Welcome_info is starting this frame...
        if Welcome_info.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Welcome_info.frameNStart = frameN  # exact frame index
            Welcome_info.tStart = t  # local t and not account for scr refresh
            Welcome_info.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Welcome_info, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Welcome_info.started')
            # update status
            Welcome_info.status = STARTED
            Welcome_info.setAutoDraw(True)
        
        # if Welcome_info is active this frame...
        if Welcome_info.status == STARTED:
            # update params
            pass
        
        # *PressEnter* updates
        waitOnFlip = False
        
        # if PressEnter is starting this frame...
        if PressEnter.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            PressEnter.frameNStart = frameN  # exact frame index
            PressEnter.tStart = t  # local t and not account for scr refresh
            PressEnter.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(PressEnter, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'PressEnter.started')
            # update status
            PressEnter.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(PressEnter.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(PressEnter.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if PressEnter.status == STARTED and not waitOnFlip:
            theseKeys = PressEnter.getKeys(keyList=["return"], ignoreKeys=["escape"], waitRelease=False)
            _PressEnter_allKeys.extend(theseKeys)
            if len(_PressEnter_allKeys):
                PressEnter.keys = _PressEnter_allKeys[-1].name  # just the last key pressed
                PressEnter.rt = _PressEnter_allKeys[-1].rt
                PressEnter.duration = _PressEnter_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *PressEnterToContinue* updates
        
        # if PressEnterToContinue is starting this frame...
        if PressEnterToContinue.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            PressEnterToContinue.frameNStart = frameN  # exact frame index
            PressEnterToContinue.tStart = t  # local t and not account for scr refresh
            PressEnterToContinue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(PressEnterToContinue, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'PressEnterToContinue.started')
            # update status
            PressEnterToContinue.status = STARTED
            PressEnterToContinue.setAutoDraw(True)
        
        # if PressEnterToContinue is active this frame...
        if PressEnterToContinue.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Welcome,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Welcome.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Welcome.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Welcome" ---
    for thisComponent in Welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Welcome
    Welcome.tStop = globalClock.getTime(format='float')
    Welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Welcome.stopped', Welcome.tStop)
    # check responses
    if PressEnter.keys in ['', [], None]:  # No response was made
        PressEnter.keys = None
    thisExp.addData('PressEnter.keys',PressEnter.keys)
    if PressEnter.keys != None:  # we had a response
        thisExp.addData('PressEnter.rt', PressEnter.rt)
        thisExp.addData('PressEnter.duration', PressEnter.duration)
    
    
    thisExp.nextEntry()
    # the Routine "Welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=5.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('ExcelOutputDraft.xlsx'), 
        seed=3, 
        isTrials=True, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "initiate" ---
        # create an object to store info about Routine initiate
        initiate = data.Routine(
            name='initiate',
            components=[],
        )
        initiate.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        img_Var = image_path  # Set Routine start values for img_Var
        thisExp.addData('img_Var.routineStartVal', img_Var)  # Save exp start value
        logical_Var = logical  # Set Routine start values for logical_Var
        corrColor_Var = corrColor  # Set Routine start values for corrColor_Var
        corrR_Var = corrR  # Set Routine start values for corrR_Var
        corrG_Var = corrG  # Set Routine start values for corrG_Var
        corrB_Var = corrB  # Set Routine start values for corrB_Var
        filterCount = 0  # Set Routine start values for filterCount
        img_Var1 = img_Var  # Set Routine start values for img_Var1
        # Run 'Begin Routine' code from code
        startTrialTime = globalClock.getTime(format='float')
        trial_counter += 1
        # store start times for initiate
        initiate.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        initiate.tStart = globalClock.getTime(format='float')
        initiate.status = STARTED
        thisExp.addData('initiate.started', initiate.tStart)
        initiate.maxDuration = 1
        # skip Routine initiate if its 'Skip if' condition is True
        initiate.skipped = continueRoutine and not (BreakIt == True)
        continueRoutine = initiate.skipped
        # keep track of which components have finished
        initiateComponents = initiate.components
        for thisComponent in initiate.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "initiate" ---
        thisExp.currentRoutine = initiate
        initiate.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > initiate.maxDuration-frameTolerance:
                initiate.maxDurationReached = True
                continueRoutine = False
            # Run 'Each Frame' code from code
            if event.getKeys(['p']):
                BreakIt = True
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=initiate,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                initiate.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if initiate.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in initiate.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "initiate" ---
        for thisComponent in initiate.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for initiate
        initiate.tStop = globalClock.getTime(format='float')
        initiate.tStopRefresh = tThisFlipGlobal
        thisExp.addData('initiate.stopped', initiate.tStop)
        thisExp.addData('img_Var.routineEndVal', img_Var)  # Save end Routine value
        thisExp.addData('logical_Var.routineEndVal', logical_Var)  # Save end Routine value
        thisExp.addData('corrColor_Var.routineEndVal', corrColor_Var)  # Save end Routine value
        thisExp.addData('corrR_Var.routineEndVal', corrR_Var)  # Save end Routine value
        thisExp.addData('corrG_Var.routineEndVal', corrG_Var)  # Save end Routine value
        thisExp.addData('corrB_Var.routineEndVal', corrB_Var)  # Save end Routine value
        thisExp.addData('filterCount.routineEndVal', filterCount)  # Save end Routine value
        thisExp.addData('img_Var1.routineEndVal', img_Var1)  # Save end Routine value
        # Run 'End Routine' code from code
        thisExp.addData('trial_counter.routineEndVal', trial_counter)  # Save end Routine value
        # the Routine "initiate" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial_1" ---
        # create an object to store info about Routine trial_1
        trial_1 = data.Routine(
            name='trial_1',
            components=[img, spacebar, blue, yellow, purple, cyan, red, green, Instructions, RBGInfo, R, G, B],
        )
        trial_1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        img.setColor('white', colorSpace='rgb')
        img.setOpacity(1.0)
        img.setPos((-500, 0))
        img.setSize((800,800))
        img.setOri(0.0)
        # Run 'Begin Routine' code from changeImgName
        filterChange = 0
        oldFilterChange = 0
        filterCountContainer = []
        oldAnswer = ""
        answerContainer = []
        # create starting attributes for spacebar
        spacebar.keys = []
        spacebar.rt = []
        _spacebar_allKeys = []
        blue.setText('Blue')
        # reset blue to account for continued clicks & clear times on/off
        blue.reset()
        yellow.setText('Yellow')
        # reset yellow to account for continued clicks & clear times on/off
        yellow.reset()
        purple.setText('Purple')
        # reset purple to account for continued clicks & clear times on/off
        purple.reset()
        cyan.setText('Cyan')
        # reset cyan to account for continued clicks & clear times on/off
        cyan.reset()
        red.setText('Red')
        # reset red to account for continued clicks & clear times on/off
        red.reset()
        green.setText('Green')
        # reset green to account for continued clicks & clear times on/off
        green.reset()
        Instructions.setText('Click the spacebar to shuffle through the filters to recognize the color.\nPress the button of the color you think it corresponds to.')
        answer = ""  # Set Routine start values for answer
        thisExp.addData('answer.routineStartVal', answer)  # Save exp start value
        R.setText(corrR_Var)
        G.setText(corrG_Var)
        B.setText(corrB_Var)
        # store start times for trial_1
        trial_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_1.tStart = globalClock.getTime(format='float')
        trial_1.status = STARTED
        thisExp.addData('trial_1.started', trial_1.tStart)
        # skip Routine trial_1 if its 'Skip if' condition is True
        trial_1.skipped = continueRoutine and not (BreakIt == True)
        continueRoutine = trial_1.skipped
        # keep track of which components have finished
        trial_1Components = trial_1.components
        for thisComponent in trial_1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_1" ---
        thisExp.currentRoutine = trial_1
        trial_1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *img* updates
            
            # if img is starting this frame...
            if img.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                img.frameNStart = frameN  # exact frame index
                img.tStart = t  # local t and not account for scr refresh
                img.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img.started')
                # update status
                img.status = STARTED
                img.setAutoDraw(True)
            
            # if img is active this frame...
            if img.status == STARTED:
                # update params
                img.setImage(img_Var1, log=False)
            # Run 'Each Frame' code from changeImgName
            if event.getKeys(['space']):
                filterChange += 1
                if oldFilterChange != filterChange:
                    if filterChange == 4: 
                       filterChange = 0
                       filterCount += 1
                    # CHANGE the picture
                    img_Var1 = img_Var[0:len(img_Var)-6] + "_" + str(filterChange) + ".png"
                    oldFilterChange = filterChange
                    filterCountContainer.append(routineTimer.getTime())
            if event.getKeys(['p']):
                BreakIt = True
            if oldAnswer != answer:
                answerContainer.append([answer,globalClock.getTime(format='float')])
                oldAnswer = answer
            
            # *spacebar* updates
            waitOnFlip = False
            
            # if spacebar is starting this frame...
            if spacebar.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
                # keep track of start time/frame for later
                spacebar.frameNStart = frameN  # exact frame index
                spacebar.tStart = t  # local t and not account for scr refresh
                spacebar.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(spacebar, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'spacebar.started')
                # update status
                spacebar.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(spacebar.clock.reset)  # t=0 on next screen flip
            if spacebar.status == STARTED and not waitOnFlip:
                theseKeys = spacebar.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _spacebar_allKeys.extend(theseKeys)
                if len(_spacebar_allKeys):
                    spacebar.keys = [key.name for key in _spacebar_allKeys]  # storing all keys
                    spacebar.rt = [key.rt for key in _spacebar_allKeys]
                    spacebar.duration = [key.duration for key in _spacebar_allKeys]
            # *blue* updates
            
            # if blue is starting this frame...
            if blue.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                blue.frameNStart = frameN  # exact frame index
                blue.tStart = t  # local t and not account for scr refresh
                blue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blue.started')
                # update status
                blue.status = STARTED
                win.callOnFlip(blue.buttonClock.reset)
                blue.setAutoDraw(True)
            
            # if blue is active this frame...
            if blue.status == STARTED:
                # update params
                pass
                # check whether blue has been pressed
                if blue.isClicked:
                    if not blue.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        blue.timesOn.append(routineTimer.getTime())
                        blue.timesOff.append(routineTimer.getTime())
                    elif len(blue.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        blue.timesOff[-1] = routineTimer.getTime()
                    if not blue.wasClicked:
                        # end routine when blue is clicked
                        continueRoutine = False
                    if not blue.wasClicked:
                        # run callback code when blue is clicked
                        answer = "Blue"
                        colorList.append(answer)
            # take note of whether blue was clicked, so that next frame we know if clicks are new
            blue.wasClicked = blue.isClicked and blue.status == STARTED
            # *yellow* updates
            
            # if yellow is starting this frame...
            if yellow.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                yellow.frameNStart = frameN  # exact frame index
                yellow.tStart = t  # local t and not account for scr refresh
                yellow.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(yellow, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'yellow.started')
                # update status
                yellow.status = STARTED
                win.callOnFlip(yellow.buttonClock.reset)
                yellow.setAutoDraw(True)
            
            # if yellow is active this frame...
            if yellow.status == STARTED:
                # update params
                pass
                # check whether yellow has been pressed
                if yellow.isClicked:
                    if not yellow.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        yellow.timesOn.append(routineTimer.getTime())
                        yellow.timesOff.append(routineTimer.getTime())
                    elif len(yellow.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        yellow.timesOff[-1] = routineTimer.getTime()
                    if not yellow.wasClicked:
                        # end routine when yellow is clicked
                        continueRoutine = False
                    if not yellow.wasClicked:
                        # run callback code when yellow is clicked
                        answer = "Yellow"
                        colorList.append(answer)
            # take note of whether yellow was clicked, so that next frame we know if clicks are new
            yellow.wasClicked = yellow.isClicked and yellow.status == STARTED
            # *purple* updates
            
            # if purple is starting this frame...
            if purple.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                purple.frameNStart = frameN  # exact frame index
                purple.tStart = t  # local t and not account for scr refresh
                purple.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(purple, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'purple.started')
                # update status
                purple.status = STARTED
                win.callOnFlip(purple.buttonClock.reset)
                purple.setAutoDraw(True)
            
            # if purple is active this frame...
            if purple.status == STARTED:
                # update params
                pass
                # check whether purple has been pressed
                if purple.isClicked:
                    if not purple.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        purple.timesOn.append(routineTimer.getTime())
                        purple.timesOff.append(routineTimer.getTime())
                    elif len(purple.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        purple.timesOff[-1] = routineTimer.getTime()
                    if not purple.wasClicked:
                        # end routine when purple is clicked
                        continueRoutine = False
                    if not purple.wasClicked:
                        # run callback code when purple is clicked
                        answer = "Purple"
                        colorList.append(answer)
            # take note of whether purple was clicked, so that next frame we know if clicks are new
            purple.wasClicked = purple.isClicked and purple.status == STARTED
            # *cyan* updates
            
            # if cyan is starting this frame...
            if cyan.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                cyan.frameNStart = frameN  # exact frame index
                cyan.tStart = t  # local t and not account for scr refresh
                cyan.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cyan, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cyan.started')
                # update status
                cyan.status = STARTED
                win.callOnFlip(cyan.buttonClock.reset)
                cyan.setAutoDraw(True)
            
            # if cyan is active this frame...
            if cyan.status == STARTED:
                # update params
                pass
                # check whether cyan has been pressed
                if cyan.isClicked:
                    if not cyan.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        cyan.timesOn.append(routineTimer.getTime())
                        cyan.timesOff.append(routineTimer.getTime())
                    elif len(cyan.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        cyan.timesOff[-1] = routineTimer.getTime()
                    if not cyan.wasClicked:
                        # end routine when cyan is clicked
                        continueRoutine = False
                    if not cyan.wasClicked:
                        # run callback code when cyan is clicked
                        answer = "Cyan"
                        colorList.append(answer)
            # take note of whether cyan was clicked, so that next frame we know if clicks are new
            cyan.wasClicked = cyan.isClicked and cyan.status == STARTED
            # *red* updates
            
            # if red is starting this frame...
            if red.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                red.frameNStart = frameN  # exact frame index
                red.tStart = t  # local t and not account for scr refresh
                red.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(red, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'red.started')
                # update status
                red.status = STARTED
                win.callOnFlip(red.buttonClock.reset)
                red.setAutoDraw(True)
            
            # if red is active this frame...
            if red.status == STARTED:
                # update params
                pass
                # check whether red has been pressed
                if red.isClicked:
                    if not red.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        red.timesOn.append(routineTimer.getTime())
                        red.timesOff.append(routineTimer.getTime())
                    elif len(red.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        red.timesOff[-1] = routineTimer.getTime()
                    if not red.wasClicked:
                        # end routine when red is clicked
                        continueRoutine = False
                    if not red.wasClicked:
                        # run callback code when red is clicked
                        answer = "Red"
                        colorList.append(answer)
            # take note of whether red was clicked, so that next frame we know if clicks are new
            red.wasClicked = red.isClicked and red.status == STARTED
            # *green* updates
            
            # if green is starting this frame...
            if green.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                green.frameNStart = frameN  # exact frame index
                green.tStart = t  # local t and not account for scr refresh
                green.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(green, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'green.started')
                # update status
                green.status = STARTED
                win.callOnFlip(green.buttonClock.reset)
                green.setAutoDraw(True)
            
            # if green is active this frame...
            if green.status == STARTED:
                # update params
                pass
                # check whether green has been pressed
                if green.isClicked:
                    if not green.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        green.timesOn.append(routineTimer.getTime())
                        green.timesOff.append(routineTimer.getTime())
                    elif len(green.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        green.timesOff[-1] = routineTimer.getTime()
                    if not green.wasClicked:
                        # end routine when green is clicked
                        continueRoutine = False
                    if not green.wasClicked:
                        # run callback code when green is clicked
                        answer = "Green"
                        colorList.append(answer)
            # take note of whether green was clicked, so that next frame we know if clicks are new
            green.wasClicked = green.isClicked and green.status == STARTED
            
            # *Instructions* updates
            
            # if Instructions is starting this frame...
            if Instructions.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Instructions.frameNStart = frameN  # exact frame index
                Instructions.tStart = t  # local t and not account for scr refresh
                Instructions.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Instructions, 'tStartRefresh')  # time at next scr refresh
                # update status
                Instructions.status = STARTED
                Instructions.setAutoDraw(True)
            
            # if Instructions is active this frame...
            if Instructions.status == STARTED:
                # update params
                pass
            
            # *RBGInfo* updates
            
            # if RBGInfo is starting this frame...
            if RBGInfo.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                RBGInfo.frameNStart = frameN  # exact frame index
                RBGInfo.tStart = t  # local t and not account for scr refresh
                RBGInfo.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(RBGInfo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'RBGInfo.started')
                # update status
                RBGInfo.status = STARTED
                RBGInfo.setAutoDraw(True)
            
            # if RBGInfo is active this frame...
            if RBGInfo.status == STARTED:
                # update params
                pass
            
            # *R* updates
            
            # if R is starting this frame...
            if R.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                R.frameNStart = frameN  # exact frame index
                R.tStart = t  # local t and not account for scr refresh
                R.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(R, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'R.started')
                # update status
                R.status = STARTED
                R.setAutoDraw(True)
            
            # if R is active this frame...
            if R.status == STARTED:
                # update params
                pass
            
            # *G* updates
            
            # if G is starting this frame...
            if G.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                G.frameNStart = frameN  # exact frame index
                G.tStart = t  # local t and not account for scr refresh
                G.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(G, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'G.started')
                # update status
                G.status = STARTED
                G.setAutoDraw(True)
            
            # if G is active this frame...
            if G.status == STARTED:
                # update params
                pass
            
            # *B* updates
            
            # if B is starting this frame...
            if B.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                B.frameNStart = frameN  # exact frame index
                B.tStart = t  # local t and not account for scr refresh
                B.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(B, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'B.started')
                # update status
                B.status = STARTED
                B.setAutoDraw(True)
            
            # if B is active this frame...
            if B.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial_1,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                trial_1.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if trial_1.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in trial_1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_1" ---
        for thisComponent in trial_1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_1
        trial_1.tStop = globalClock.getTime(format='float')
        trial_1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_1.stopped', trial_1.tStop)
        # Run 'End Routine' code from changeImgName
        thisExp.addData('corrColorVar.routineEndVal', corrColor_Var)  # Save end Routine value
        thisExp.addData('filterChange.routineEndVal', filterChange)  # Save end Routine value
        thisExp.addData('filterCount.routineEndVal', filterCount)  # Save end Routine value
        thisExp.addData('filterCount.routineEndVal', answerContainer)  # Save end Routine value
        thisExp.addData('filterCount.routineEndVal', filterCountContainer)  # Save end Routine value
        # check responses
        if spacebar.keys in ['', [], None]:  # No response was made
            spacebar.keys = None
        trials.addData('spacebar.keys',spacebar.keys)
        if spacebar.keys != None:  # we had a response
            trials.addData('spacebar.rt', spacebar.rt)
            trials.addData('spacebar.duration', spacebar.duration)
        trials.addData('blue.numClicks', blue.numClicks)
        if blue.numClicks:
           trials.addData('blue.timesOn', blue.timesOn)
           trials.addData('blue.timesOff', blue.timesOff)
        else:
           trials.addData('blue.timesOn', "")
           trials.addData('blue.timesOff', "")
        trials.addData('yellow.numClicks', yellow.numClicks)
        if yellow.numClicks:
           trials.addData('yellow.timesOn', yellow.timesOn)
           trials.addData('yellow.timesOff', yellow.timesOff)
        else:
           trials.addData('yellow.timesOn', "")
           trials.addData('yellow.timesOff', "")
        trials.addData('purple.numClicks', purple.numClicks)
        if purple.numClicks:
           trials.addData('purple.timesOn', purple.timesOn)
           trials.addData('purple.timesOff', purple.timesOff)
        else:
           trials.addData('purple.timesOn', "")
           trials.addData('purple.timesOff', "")
        trials.addData('cyan.numClicks', cyan.numClicks)
        if cyan.numClicks:
           trials.addData('cyan.timesOn', cyan.timesOn)
           trials.addData('cyan.timesOff', cyan.timesOff)
        else:
           trials.addData('cyan.timesOn', "")
           trials.addData('cyan.timesOff', "")
        trials.addData('red.numClicks', red.numClicks)
        if red.numClicks:
           trials.addData('red.timesOn', red.timesOn)
           trials.addData('red.timesOff', red.timesOff)
        else:
           trials.addData('red.timesOn', "")
           trials.addData('red.timesOff', "")
        trials.addData('green.numClicks', green.numClicks)
        if green.numClicks:
           trials.addData('green.timesOn', green.timesOn)
           trials.addData('green.timesOff', green.timesOff)
        else:
           trials.addData('green.timesOn', "")
           trials.addData('green.timesOff', "")
        thisExp.addData('answer.routineEndVal', answer)  # Save end Routine value
        # the Routine "trial_1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Pause350" ---
        # create an object to store info about Routine Pause350
        Pause350 = data.Routine(
            name='Pause350',
            components=[text_8],
        )
        Pause350.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Pause350
        Pause350.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Pause350.tStart = globalClock.getTime(format='float')
        Pause350.status = STARTED
        thisExp.addData('Pause350.started', Pause350.tStart)
        Pause350.maxDuration = None
        # skip Routine Pause350 if its 'Skip if' condition is True
        Pause350.skipped = continueRoutine and not (BreakIt == True)
        continueRoutine = Pause350.skipped
        # keep track of which components have finished
        Pause350Components = Pause350.components
        for thisComponent in Pause350.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Pause350" ---
        thisExp.currentRoutine = Pause350
        Pause350.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.35:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_8* updates
            
            # if text_8 is starting this frame...
            if text_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_8.frameNStart = frameN  # exact frame index
                text_8.tStart = t  # local t and not account for scr refresh
                text_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_8.started')
                # update status
                text_8.status = STARTED
                text_8.setAutoDraw(True)
            
            # if text_8 is active this frame...
            if text_8.status == STARTED:
                # update params
                pass
            
            # if text_8 is stopping this frame...
            if text_8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_8.tStartRefresh + 0.35-frameTolerance:
                    # keep track of stop time/frame for later
                    text_8.tStop = t  # not accounting for scr refresh
                    text_8.tStopRefresh = tThisFlipGlobal  # on global time
                    text_8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_8.stopped')
                    # update status
                    text_8.status = FINISHED
                    text_8.setAutoDraw(False)
            # Run 'Each Frame' code from code_3
            if event.getKeys(['p']):
                BreakIt = True
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Pause350,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Pause350.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Pause350.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Pause350.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Pause350" ---
        for thisComponent in Pause350.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Pause350
        Pause350.tStop = globalClock.getTime(format='float')
        Pause350.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Pause350.stopped', Pause350.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Pause350.maxDurationReached:
            routineTimer.addTime(-Pause350.maxDuration)
        elif Pause350.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.350000)
        
        # --- Prepare to start Routine "confidence" ---
        # create an object to store info about Routine confidence
        confidence = data.Routine(
            name='confidence',
            components=[slider, text_9, mouse_2, button2, text_answer, blue_3, yellow_3, purple_3, cyan_3, red_3, green_3],
        )
        confidence.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        slider.reset()
        # setup some python lists for storing info about the mouse_2
        mouse_2.x = []
        mouse_2.y = []
        mouse_2.leftButton = []
        mouse_2.midButton = []
        mouse_2.rightButton = []
        mouse_2.time = []
        mouse_2.clicked_name = []
        gotValidClick = False  # until a click is received
        button2.setText('Click here to continue.')
        # reset button2 to account for continued clicks & clear times on/off
        button2.reset()
        blue_3.setText('Blue')
        # reset blue_3 to account for continued clicks & clear times on/off
        blue_3.reset()
        yellow_3.setText('Yellow')
        # reset yellow_3 to account for continued clicks & clear times on/off
        yellow_3.reset()
        purple_3.setText('Purple')
        # reset purple_3 to account for continued clicks & clear times on/off
        purple_3.reset()
        cyan_3.setText('Cyan')
        # reset cyan_3 to account for continued clicks & clear times on/off
        cyan_3.reset()
        red_3.setText('Red')
        # reset red_3 to account for continued clicks & clear times on/off
        red_3.reset()
        green_3.setText('Green')
        # reset green_3 to account for continued clicks & clear times on/off
        green_3.reset()
        # store start times for confidence
        confidence.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        confidence.tStart = globalClock.getTime(format='float')
        confidence.status = STARTED
        thisExp.addData('confidence.started', confidence.tStart)
        confidence.maxDuration = None
        # skip Routine confidence if its 'Skip if' condition is True
        confidence.skipped = continueRoutine and not (BreakIt == True)
        continueRoutine = confidence.skipped
        # keep track of which components have finished
        confidenceComponents = confidence.components
        for thisComponent in confidence.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "confidence" ---
        thisExp.currentRoutine = confidence
        confidence.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *slider* updates
            
            # if slider is starting this frame...
            if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider.frameNStart = frameN  # exact frame index
                slider.tStart = t  # local t and not account for scr refresh
                slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider.started')
                # update status
                slider.status = STARTED
                slider.setAutoDraw(True)
            
            # if slider is active this frame...
            if slider.status == STARTED:
                # update params
                pass
            
            # *text_9* updates
            
            # if text_9 is starting this frame...
            if text_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_9.frameNStart = frameN  # exact frame index
                text_9.tStart = t  # local t and not account for scr refresh
                text_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_9.started')
                # update status
                text_9.status = STARTED
                text_9.setAutoDraw(True)
            
            # if text_9 is active this frame...
            if text_9.status == STARTED:
                # update params
                pass
            # *mouse_2* updates
            
            # if mouse_2 is starting this frame...
            if mouse_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_2.frameNStart = frameN  # exact frame index
                mouse_2.tStart = t  # local t and not account for scr refresh
                mouse_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_2.started', t)
                # update status
                mouse_2.status = STARTED
                mouse_2.mouseClock.reset()
                prevButtonState = mouse_2.getPressed()  # if button is down already this ISN'T a new click
            if mouse_2.status == STARTED:  # only update if started and not finished!
                buttons = mouse_2.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(button2, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(mouse_2):
                                gotValidClick = True
                                mouse_2.clicked_name.append(obj.name)
                        if not gotValidClick:
                            mouse_2.clicked_name.append(None)
                        x, y = mouse_2.getPos()
                        mouse_2.x.append(float(x))
                        mouse_2.y.append(float(y))
                        buttons = mouse_2.getPressed()
                        mouse_2.leftButton.append(buttons[0])
                        mouse_2.midButton.append(buttons[1])
                        mouse_2.rightButton.append(buttons[2])
                        mouse_2.time.append(mouse_2.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            # *button2* updates
            
            # if button2 is starting this frame...
            if button2.status == NOT_STARTED and slider.rating:
                # keep track of start time/frame for later
                button2.frameNStart = frameN  # exact frame index
                button2.tStart = t  # local t and not account for scr refresh
                button2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(button2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'button2.started')
                # update status
                button2.status = STARTED
                win.callOnFlip(button2.buttonClock.reset)
                button2.setAutoDraw(True)
            
            # if button2 is active this frame...
            if button2.status == STARTED:
                # update params
                pass
                # check whether button2 has been pressed
                if button2.isClicked:
                    if not button2.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        button2.timesOn.append(routineTimer.getTime())
                        button2.timesOff.append(routineTimer.getTime())
                    elif len(button2.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        button2.timesOff[-1] = routineTimer.getTime()
                    if not button2.wasClicked:
                        # end routine when button2 is clicked
                        continueRoutine = False
                    if not button2.wasClicked:
                        # run callback code when button2 is clicked
                        pass
            # take note of whether button2 was clicked, so that next frame we know if clicks are new
            button2.wasClicked = button2.isClicked and button2.status == STARTED
            
            # *text_answer* updates
            
            # if text_answer is starting this frame...
            if text_answer.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_answer.frameNStart = frameN  # exact frame index
                text_answer.tStart = t  # local t and not account for scr refresh
                text_answer.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_answer, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_answer.started')
                # update status
                text_answer.status = STARTED
                text_answer.setAutoDraw(True)
            
            # if text_answer is active this frame...
            if text_answer.status == STARTED:
                # update params
                text_answer.setText(answer, log=False)
            # *blue_3* updates
            
            # if blue_3 is starting this frame...
            if blue_3.status == NOT_STARTED and answer == "Blue":
                # keep track of start time/frame for later
                blue_3.frameNStart = frameN  # exact frame index
                blue_3.tStart = t  # local t and not account for scr refresh
                blue_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blue_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blue_3.started')
                # update status
                blue_3.status = STARTED
                win.callOnFlip(blue_3.buttonClock.reset)
                blue_3.setAutoDraw(True)
            
            # if blue_3 is active this frame...
            if blue_3.status == STARTED:
                # update params
                pass
                # check whether blue_3 has been pressed
                if blue_3.isClicked:
                    if not blue_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        blue_3.timesOn.append(routineTimer.getTime())
                        blue_3.timesOff.append(routineTimer.getTime())
                    elif len(blue_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        blue_3.timesOff[-1] = routineTimer.getTime()
                    # run callback code when blue_3 is clicked
                    pass
            # take note of whether blue_3 was clicked, so that next frame we know if clicks are new
            blue_3.wasClicked = blue_3.isClicked and blue_3.status == STARTED
            # *yellow_3* updates
            
            # if yellow_3 is starting this frame...
            if yellow_3.status == NOT_STARTED and answer == "Yellow":
                # keep track of start time/frame for later
                yellow_3.frameNStart = frameN  # exact frame index
                yellow_3.tStart = t  # local t and not account for scr refresh
                yellow_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(yellow_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'yellow_3.started')
                # update status
                yellow_3.status = STARTED
                win.callOnFlip(yellow_3.buttonClock.reset)
                yellow_3.setAutoDraw(True)
            
            # if yellow_3 is active this frame...
            if yellow_3.status == STARTED:
                # update params
                pass
                # check whether yellow_3 has been pressed
                if yellow_3.isClicked:
                    if not yellow_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        yellow_3.timesOn.append(routineTimer.getTime())
                        yellow_3.timesOff.append(routineTimer.getTime())
                    elif len(yellow_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        yellow_3.timesOff[-1] = routineTimer.getTime()
                    # run callback code when yellow_3 is clicked
                    pass
            # take note of whether yellow_3 was clicked, so that next frame we know if clicks are new
            yellow_3.wasClicked = yellow_3.isClicked and yellow_3.status == STARTED
            # *purple_3* updates
            
            # if purple_3 is starting this frame...
            if purple_3.status == NOT_STARTED and answer == "Purple":
                # keep track of start time/frame for later
                purple_3.frameNStart = frameN  # exact frame index
                purple_3.tStart = t  # local t and not account for scr refresh
                purple_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(purple_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'purple_3.started')
                # update status
                purple_3.status = STARTED
                win.callOnFlip(purple_3.buttonClock.reset)
                purple_3.setAutoDraw(True)
            
            # if purple_3 is active this frame...
            if purple_3.status == STARTED:
                # update params
                pass
                # check whether purple_3 has been pressed
                if purple_3.isClicked:
                    if not purple_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        purple_3.timesOn.append(routineTimer.getTime())
                        purple_3.timesOff.append(routineTimer.getTime())
                    elif len(purple_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        purple_3.timesOff[-1] = routineTimer.getTime()
                    # run callback code when purple_3 is clicked
                    pass
            # take note of whether purple_3 was clicked, so that next frame we know if clicks are new
            purple_3.wasClicked = purple_3.isClicked and purple_3.status == STARTED
            # *cyan_3* updates
            
            # if cyan_3 is starting this frame...
            if cyan_3.status == NOT_STARTED and answer == "Cyan":
                # keep track of start time/frame for later
                cyan_3.frameNStart = frameN  # exact frame index
                cyan_3.tStart = t  # local t and not account for scr refresh
                cyan_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cyan_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cyan_3.started')
                # update status
                cyan_3.status = STARTED
                win.callOnFlip(cyan_3.buttonClock.reset)
                cyan_3.setAutoDraw(True)
            
            # if cyan_3 is active this frame...
            if cyan_3.status == STARTED:
                # update params
                pass
                # check whether cyan_3 has been pressed
                if cyan_3.isClicked:
                    if not cyan_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        cyan_3.timesOn.append(routineTimer.getTime())
                        cyan_3.timesOff.append(routineTimer.getTime())
                    elif len(cyan_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        cyan_3.timesOff[-1] = routineTimer.getTime()
                    # run callback code when cyan_3 is clicked
                    pass
            # take note of whether cyan_3 was clicked, so that next frame we know if clicks are new
            cyan_3.wasClicked = cyan_3.isClicked and cyan_3.status == STARTED
            # *red_3* updates
            
            # if red_3 is starting this frame...
            if red_3.status == NOT_STARTED and answer == "Red":
                # keep track of start time/frame for later
                red_3.frameNStart = frameN  # exact frame index
                red_3.tStart = t  # local t and not account for scr refresh
                red_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(red_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'red_3.started')
                # update status
                red_3.status = STARTED
                win.callOnFlip(red_3.buttonClock.reset)
                red_3.setAutoDraw(True)
            
            # if red_3 is active this frame...
            if red_3.status == STARTED:
                # update params
                pass
                # check whether red_3 has been pressed
                if red_3.isClicked:
                    if not red_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        red_3.timesOn.append(routineTimer.getTime())
                        red_3.timesOff.append(routineTimer.getTime())
                    elif len(red_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        red_3.timesOff[-1] = routineTimer.getTime()
                    # run callback code when red_3 is clicked
                    pass
            # take note of whether red_3 was clicked, so that next frame we know if clicks are new
            red_3.wasClicked = red_3.isClicked and red_3.status == STARTED
            # *green_3* updates
            
            # if green_3 is starting this frame...
            if green_3.status == NOT_STARTED and answer == "Green":
                # keep track of start time/frame for later
                green_3.frameNStart = frameN  # exact frame index
                green_3.tStart = t  # local t and not account for scr refresh
                green_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(green_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'green_3.started')
                # update status
                green_3.status = STARTED
                win.callOnFlip(green_3.buttonClock.reset)
                green_3.setAutoDraw(True)
            
            # if green_3 is active this frame...
            if green_3.status == STARTED:
                # update params
                pass
                # check whether green_3 has been pressed
                if green_3.isClicked:
                    if not green_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        green_3.timesOn.append(routineTimer.getTime())
                        green_3.timesOff.append(routineTimer.getTime())
                    elif len(green_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        green_3.timesOff[-1] = routineTimer.getTime()
                    # run callback code when green_3 is clicked
                    pass
            # take note of whether green_3 was clicked, so that next frame we know if clicks are new
            green_3.wasClicked = green_3.isClicked and green_3.status == STARTED
            # Run 'Each Frame' code from code_2
            if event.getKeys(['p']):
                BreakIt == True
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=confidence,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                confidence.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if confidence.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in confidence.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "confidence" ---
        for thisComponent in confidence.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for confidence
        confidence.tStop = globalClock.getTime(format='float')
        confidence.tStopRefresh = tThisFlipGlobal
        thisExp.addData('confidence.stopped', confidence.tStop)
        trials.addData('slider.response', slider.getRating())
        trials.addData('slider.rt', slider.getRT())
        trials.addData('slider.history', slider.getHistory())
        # store data for trials (TrialHandler)
        trials.addData('mouse_2.x', mouse_2.x)
        trials.addData('mouse_2.y', mouse_2.y)
        trials.addData('mouse_2.leftButton', mouse_2.leftButton)
        trials.addData('mouse_2.midButton', mouse_2.midButton)
        trials.addData('mouse_2.rightButton', mouse_2.rightButton)
        trials.addData('mouse_2.time', mouse_2.time)
        trials.addData('mouse_2.clicked_name', mouse_2.clicked_name)
        trials.addData('button2.numClicks', button2.numClicks)
        if button2.numClicks:
           trials.addData('button2.timesOn', button2.timesOn)
           trials.addData('button2.timesOff', button2.timesOff)
        else:
           trials.addData('button2.timesOn', "")
           trials.addData('button2.timesOff', "")
        trials.addData('blue_3.numClicks', blue_3.numClicks)
        if blue_3.numClicks:
           trials.addData('blue_3.timesOn', blue_3.timesOn)
           trials.addData('blue_3.timesOff', blue_3.timesOff)
        else:
           trials.addData('blue_3.timesOn', "")
           trials.addData('blue_3.timesOff', "")
        trials.addData('yellow_3.numClicks', yellow_3.numClicks)
        if yellow_3.numClicks:
           trials.addData('yellow_3.timesOn', yellow_3.timesOn)
           trials.addData('yellow_3.timesOff', yellow_3.timesOff)
        else:
           trials.addData('yellow_3.timesOn', "")
           trials.addData('yellow_3.timesOff', "")
        trials.addData('purple_3.numClicks', purple_3.numClicks)
        if purple_3.numClicks:
           trials.addData('purple_3.timesOn', purple_3.timesOn)
           trials.addData('purple_3.timesOff', purple_3.timesOff)
        else:
           trials.addData('purple_3.timesOn', "")
           trials.addData('purple_3.timesOff', "")
        trials.addData('cyan_3.numClicks', cyan_3.numClicks)
        if cyan_3.numClicks:
           trials.addData('cyan_3.timesOn', cyan_3.timesOn)
           trials.addData('cyan_3.timesOff', cyan_3.timesOff)
        else:
           trials.addData('cyan_3.timesOn', "")
           trials.addData('cyan_3.timesOff', "")
        trials.addData('red_3.numClicks', red_3.numClicks)
        if red_3.numClicks:
           trials.addData('red_3.timesOn', red_3.timesOn)
           trials.addData('red_3.timesOff', red_3.timesOff)
        else:
           trials.addData('red_3.timesOn', "")
           trials.addData('red_3.timesOff', "")
        trials.addData('green_3.numClicks', green_3.numClicks)
        if green_3.numClicks:
           trials.addData('green_3.timesOn', green_3.timesOn)
           trials.addData('green_3.timesOff', green_3.timesOff)
        else:
           trials.addData('green_3.timesOn', "")
           trials.addData('green_3.timesOff', "")
        # the Routine "confidence" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Pause350" ---
        # create an object to store info about Routine Pause350
        Pause350 = data.Routine(
            name='Pause350',
            components=[text_8],
        )
        Pause350.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Pause350
        Pause350.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Pause350.tStart = globalClock.getTime(format='float')
        Pause350.status = STARTED
        thisExp.addData('Pause350.started', Pause350.tStart)
        Pause350.maxDuration = None
        # skip Routine Pause350 if its 'Skip if' condition is True
        Pause350.skipped = continueRoutine and not (BreakIt == True)
        continueRoutine = Pause350.skipped
        # keep track of which components have finished
        Pause350Components = Pause350.components
        for thisComponent in Pause350.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Pause350" ---
        thisExp.currentRoutine = Pause350
        Pause350.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.35:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_8* updates
            
            # if text_8 is starting this frame...
            if text_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_8.frameNStart = frameN  # exact frame index
                text_8.tStart = t  # local t and not account for scr refresh
                text_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_8.started')
                # update status
                text_8.status = STARTED
                text_8.setAutoDraw(True)
            
            # if text_8 is active this frame...
            if text_8.status == STARTED:
                # update params
                pass
            
            # if text_8 is stopping this frame...
            if text_8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_8.tStartRefresh + 0.35-frameTolerance:
                    # keep track of stop time/frame for later
                    text_8.tStop = t  # not accounting for scr refresh
                    text_8.tStopRefresh = tThisFlipGlobal  # on global time
                    text_8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_8.stopped')
                    # update status
                    text_8.status = FINISHED
                    text_8.setAutoDraw(False)
            # Run 'Each Frame' code from code_3
            if event.getKeys(['p']):
                BreakIt = True
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Pause350,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Pause350.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Pause350.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Pause350.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Pause350" ---
        for thisComponent in Pause350.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Pause350
        Pause350.tStop = globalClock.getTime(format='float')
        Pause350.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Pause350.stopped', Pause350.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Pause350.maxDurationReached:
            routineTimer.addTime(-Pause350.maxDuration)
        elif Pause350.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.350000)
        
        # --- Prepare to start Routine "confirm" ---
        # create an object to store info about Routine confirm
        confirm = data.Routine(
            name='confirm',
            components=[Button3, correctAnswer, wrongAnswer, corrColor_2, blue_2, yellow_2, purple_2, cyan_2, red_2, green_2, RBGInfo_2, R_2, G_2, B_2, key_resp, BaselineOverrideInfo],
        )
        confirm.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # reset Button3 to account for continued clicks & clear times on/off
        Button3.reset()
        corrColor_2.setText(corrColor_Var
        )
        blue_2.setText('Blue')
        # reset blue_2 to account for continued clicks & clear times on/off
        blue_2.reset()
        yellow_2.setText('Yellow')
        # reset yellow_2 to account for continued clicks & clear times on/off
        yellow_2.reset()
        purple_2.setText('Purple')
        # reset purple_2 to account for continued clicks & clear times on/off
        purple_2.reset()
        cyan_2.setText('Cyan')
        # reset cyan_2 to account for continued clicks & clear times on/off
        cyan_2.reset()
        red_2.setText('Red')
        # reset red_2 to account for continued clicks & clear times on/off
        red_2.reset()
        green_2.setText('Green')
        # reset green_2 to account for continued clicks & clear times on/off
        green_2.reset()
        R_2.setText(corrR_Var)
        G_2.setText(corrG_Var)
        B_2.setText(corrB_Var)
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        BaselineOverrideInfo.setText('If you are exhausted from the experiment, press "P" to end it after this screen.\nOtherwise, please continue.')
        # store start times for confirm
        confirm.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        confirm.tStart = globalClock.getTime(format='float')
        confirm.status = STARTED
        thisExp.addData('confirm.started', confirm.tStart)
        confirm.maxDuration = None
        # skip Routine confirm if its 'Skip if' condition is True
        confirm.skipped = continueRoutine and not (BreakIt == True)
        continueRoutine = confirm.skipped
        # keep track of which components have finished
        confirmComponents = confirm.components
        for thisComponent in confirm.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "confirm" ---
        thisExp.currentRoutine = confirm
        confirm.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *Button3* updates
            
            # if Button3 is starting this frame...
            if Button3.status == NOT_STARTED and answer == corrColor:
                # keep track of start time/frame for later
                Button3.frameNStart = frameN  # exact frame index
                Button3.tStart = t  # local t and not account for scr refresh
                Button3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Button3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Button3.started')
                # update status
                Button3.status = STARTED
                win.callOnFlip(Button3.buttonClock.reset)
                Button3.setAutoDraw(True)
            
            # if Button3 is active this frame...
            if Button3.status == STARTED:
                # update params
                Button3.setText('Click here to continue.', log=False)
                # check whether Button3 has been pressed
                if Button3.isClicked:
                    if not Button3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        Button3.timesOn.append(routineTimer.getTime())
                        Button3.timesOff.append(routineTimer.getTime())
                    elif len(Button3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        Button3.timesOff[-1] = routineTimer.getTime()
                    if not Button3.wasClicked:
                        # end routine when Button3 is clicked
                        continueRoutine = False
                    if not Button3.wasClicked:
                        # run callback code when Button3 is clicked
                        pass
            # take note of whether Button3 was clicked, so that next frame we know if clicks are new
            Button3.wasClicked = Button3.isClicked and Button3.status == STARTED
            
            # *correctAnswer* updates
            
            # if correctAnswer is starting this frame...
            if correctAnswer.status == NOT_STARTED and answer == corrColor:
                # keep track of start time/frame for later
                correctAnswer.frameNStart = frameN  # exact frame index
                correctAnswer.tStart = t  # local t and not account for scr refresh
                correctAnswer.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(correctAnswer, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'correctAnswer.started')
                # update status
                correctAnswer.status = STARTED
                correctAnswer.setAutoDraw(True)
            
            # if correctAnswer is active this frame...
            if correctAnswer.status == STARTED:
                # update params
                correctAnswer.setText('Yay, good job!', log=False)
            
            # *wrongAnswer* updates
            
            # if wrongAnswer is starting this frame...
            if wrongAnswer.status == NOT_STARTED and answer != corrColor:
                # keep track of start time/frame for later
                wrongAnswer.frameNStart = frameN  # exact frame index
                wrongAnswer.tStart = t  # local t and not account for scr refresh
                wrongAnswer.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wrongAnswer, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wrongAnswer.started')
                # update status
                wrongAnswer.status = STARTED
                wrongAnswer.setAutoDraw(True)
            
            # if wrongAnswer is active this frame...
            if wrongAnswer.status == STARTED:
                # update params
                wrongAnswer.setText('Not quite.\n\nThe correct color was\n\n\n\n\nPress the right color to continue.', log=False)
            
            # if wrongAnswer is stopping this frame...
            if wrongAnswer.status == STARTED:
                if bool(answer == corrColor):
                    # keep track of stop time/frame for later
                    wrongAnswer.tStop = t  # not accounting for scr refresh
                    wrongAnswer.tStopRefresh = tThisFlipGlobal  # on global time
                    wrongAnswer.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'wrongAnswer.stopped')
                    # update status
                    wrongAnswer.status = FINISHED
                    wrongAnswer.setAutoDraw(False)
            
            # *corrColor_2* updates
            
            # if corrColor_2 is starting this frame...
            if corrColor_2.status == NOT_STARTED and answer != corrColor:
                # keep track of start time/frame for later
                corrColor_2.frameNStart = frameN  # exact frame index
                corrColor_2.tStart = t  # local t and not account for scr refresh
                corrColor_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(corrColor_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'corrColor_2.started')
                # update status
                corrColor_2.status = STARTED
                corrColor_2.setAutoDraw(True)
            
            # if corrColor_2 is active this frame...
            if corrColor_2.status == STARTED:
                # update params
                pass
            
            # if corrColor_2 is stopping this frame...
            if corrColor_2.status == STARTED:
                if bool(answer == corrColor):
                    # keep track of stop time/frame for later
                    corrColor_2.tStop = t  # not accounting for scr refresh
                    corrColor_2.tStopRefresh = tThisFlipGlobal  # on global time
                    corrColor_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'corrColor_2.stopped')
                    # update status
                    corrColor_2.status = FINISHED
                    corrColor_2.setAutoDraw(False)
            # *blue_2* updates
            
            # if blue_2 is starting this frame...
            if blue_2.status == NOT_STARTED and answer != corrColor:
                # keep track of start time/frame for later
                blue_2.frameNStart = frameN  # exact frame index
                blue_2.tStart = t  # local t and not account for scr refresh
                blue_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blue_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blue_2.started')
                # update status
                blue_2.status = STARTED
                win.callOnFlip(blue_2.buttonClock.reset)
                blue_2.setAutoDraw(True)
            
            # if blue_2 is active this frame...
            if blue_2.status == STARTED:
                # update params
                pass
                # check whether blue_2 has been pressed
                if blue_2.isClicked:
                    if not blue_2.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        blue_2.timesOn.append(routineTimer.getTime())
                        blue_2.timesOff.append(routineTimer.getTime())
                    elif len(blue_2.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        blue_2.timesOff[-1] = routineTimer.getTime()
                    if not blue_2.wasClicked:
                        # run callback code when blue_2 is clicked
                        answer = "Blue"
                        colorList.append(answer)
            # take note of whether blue_2 was clicked, so that next frame we know if clicks are new
            blue_2.wasClicked = blue_2.isClicked and blue_2.status == STARTED
            
            # if blue_2 is stopping this frame...
            if blue_2.status == STARTED:
                if bool(answer == corrColor):
                    # keep track of stop time/frame for later
                    blue_2.tStop = t  # not accounting for scr refresh
                    blue_2.tStopRefresh = tThisFlipGlobal  # on global time
                    blue_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blue_2.stopped')
                    # update status
                    blue_2.status = FINISHED
                    blue_2.setAutoDraw(False)
            # *yellow_2* updates
            
            # if yellow_2 is starting this frame...
            if yellow_2.status == NOT_STARTED and answer != corrColor:
                # keep track of start time/frame for later
                yellow_2.frameNStart = frameN  # exact frame index
                yellow_2.tStart = t  # local t and not account for scr refresh
                yellow_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(yellow_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'yellow_2.started')
                # update status
                yellow_2.status = STARTED
                win.callOnFlip(yellow_2.buttonClock.reset)
                yellow_2.setAutoDraw(True)
            
            # if yellow_2 is active this frame...
            if yellow_2.status == STARTED:
                # update params
                pass
                # check whether yellow_2 has been pressed
                if yellow_2.isClicked:
                    if not yellow_2.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        yellow_2.timesOn.append(routineTimer.getTime())
                        yellow_2.timesOff.append(routineTimer.getTime())
                    elif len(yellow_2.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        yellow_2.timesOff[-1] = routineTimer.getTime()
                    if not yellow_2.wasClicked:
                        # run callback code when yellow_2 is clicked
                        answer = "Yellow"
                        colorList.append(answer)
            # take note of whether yellow_2 was clicked, so that next frame we know if clicks are new
            yellow_2.wasClicked = yellow_2.isClicked and yellow_2.status == STARTED
            
            # if yellow_2 is stopping this frame...
            if yellow_2.status == STARTED:
                if bool(answer == corrColor):
                    # keep track of stop time/frame for later
                    yellow_2.tStop = t  # not accounting for scr refresh
                    yellow_2.tStopRefresh = tThisFlipGlobal  # on global time
                    yellow_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'yellow_2.stopped')
                    # update status
                    yellow_2.status = FINISHED
                    yellow_2.setAutoDraw(False)
            # *purple_2* updates
            
            # if purple_2 is starting this frame...
            if purple_2.status == NOT_STARTED and answer != corrColor:
                # keep track of start time/frame for later
                purple_2.frameNStart = frameN  # exact frame index
                purple_2.tStart = t  # local t and not account for scr refresh
                purple_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(purple_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'purple_2.started')
                # update status
                purple_2.status = STARTED
                win.callOnFlip(purple_2.buttonClock.reset)
                purple_2.setAutoDraw(True)
            
            # if purple_2 is active this frame...
            if purple_2.status == STARTED:
                # update params
                pass
                # check whether purple_2 has been pressed
                if purple_2.isClicked:
                    if not purple_2.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        purple_2.timesOn.append(routineTimer.getTime())
                        purple_2.timesOff.append(routineTimer.getTime())
                    elif len(purple_2.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        purple_2.timesOff[-1] = routineTimer.getTime()
                    if not purple_2.wasClicked:
                        # run callback code when purple_2 is clicked
                        answer = "Purple"
                        colorList.append(answer)
            # take note of whether purple_2 was clicked, so that next frame we know if clicks are new
            purple_2.wasClicked = purple_2.isClicked and purple_2.status == STARTED
            
            # if purple_2 is stopping this frame...
            if purple_2.status == STARTED:
                if bool(answer == corrColor):
                    # keep track of stop time/frame for later
                    purple_2.tStop = t  # not accounting for scr refresh
                    purple_2.tStopRefresh = tThisFlipGlobal  # on global time
                    purple_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'purple_2.stopped')
                    # update status
                    purple_2.status = FINISHED
                    purple_2.setAutoDraw(False)
            # *cyan_2* updates
            
            # if cyan_2 is starting this frame...
            if cyan_2.status == NOT_STARTED and answer != corrColor:
                # keep track of start time/frame for later
                cyan_2.frameNStart = frameN  # exact frame index
                cyan_2.tStart = t  # local t and not account for scr refresh
                cyan_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cyan_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cyan_2.started')
                # update status
                cyan_2.status = STARTED
                win.callOnFlip(cyan_2.buttonClock.reset)
                cyan_2.setAutoDraw(True)
            
            # if cyan_2 is active this frame...
            if cyan_2.status == STARTED:
                # update params
                pass
                # check whether cyan_2 has been pressed
                if cyan_2.isClicked:
                    if not cyan_2.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        cyan_2.timesOn.append(routineTimer.getTime())
                        cyan_2.timesOff.append(routineTimer.getTime())
                    elif len(cyan_2.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        cyan_2.timesOff[-1] = routineTimer.getTime()
                    if not cyan_2.wasClicked:
                        # run callback code when cyan_2 is clicked
                        answer = "Cyan"
                        colorList.append(answer)
            # take note of whether cyan_2 was clicked, so that next frame we know if clicks are new
            cyan_2.wasClicked = cyan_2.isClicked and cyan_2.status == STARTED
            
            # if cyan_2 is stopping this frame...
            if cyan_2.status == STARTED:
                if bool(answer == corrColor):
                    # keep track of stop time/frame for later
                    cyan_2.tStop = t  # not accounting for scr refresh
                    cyan_2.tStopRefresh = tThisFlipGlobal  # on global time
                    cyan_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cyan_2.stopped')
                    # update status
                    cyan_2.status = FINISHED
                    cyan_2.setAutoDraw(False)
            # *red_2* updates
            
            # if red_2 is starting this frame...
            if red_2.status == NOT_STARTED and answer != corrColor:
                # keep track of start time/frame for later
                red_2.frameNStart = frameN  # exact frame index
                red_2.tStart = t  # local t and not account for scr refresh
                red_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(red_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'red_2.started')
                # update status
                red_2.status = STARTED
                win.callOnFlip(red_2.buttonClock.reset)
                red_2.setAutoDraw(True)
            
            # if red_2 is active this frame...
            if red_2.status == STARTED:
                # update params
                pass
                # check whether red_2 has been pressed
                if red_2.isClicked:
                    if not red_2.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        red_2.timesOn.append(routineTimer.getTime())
                        red_2.timesOff.append(routineTimer.getTime())
                    elif len(red_2.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        red_2.timesOff[-1] = routineTimer.getTime()
                    if not red_2.wasClicked:
                        # run callback code when red_2 is clicked
                        answer = "Red"
                        colorList.append(answer)
            # take note of whether red_2 was clicked, so that next frame we know if clicks are new
            red_2.wasClicked = red_2.isClicked and red_2.status == STARTED
            
            # if red_2 is stopping this frame...
            if red_2.status == STARTED:
                if bool(answer == corrColor):
                    # keep track of stop time/frame for later
                    red_2.tStop = t  # not accounting for scr refresh
                    red_2.tStopRefresh = tThisFlipGlobal  # on global time
                    red_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'red_2.stopped')
                    # update status
                    red_2.status = FINISHED
                    red_2.setAutoDraw(False)
            # *green_2* updates
            
            # if green_2 is starting this frame...
            if green_2.status == NOT_STARTED and answer != corrColor:
                # keep track of start time/frame for later
                green_2.frameNStart = frameN  # exact frame index
                green_2.tStart = t  # local t and not account for scr refresh
                green_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(green_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'green_2.started')
                # update status
                green_2.status = STARTED
                win.callOnFlip(green_2.buttonClock.reset)
                green_2.setAutoDraw(True)
            
            # if green_2 is active this frame...
            if green_2.status == STARTED:
                # update params
                pass
                # check whether green_2 has been pressed
                if green_2.isClicked:
                    if not green_2.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        green_2.timesOn.append(routineTimer.getTime())
                        green_2.timesOff.append(routineTimer.getTime())
                    elif len(green_2.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        green_2.timesOff[-1] = routineTimer.getTime()
                    if not green_2.wasClicked:
                        # run callback code when green_2 is clicked
                        answer = "Green"
                        colorList.append(answer)
            # take note of whether green_2 was clicked, so that next frame we know if clicks are new
            green_2.wasClicked = green_2.isClicked and green_2.status == STARTED
            
            # if green_2 is stopping this frame...
            if green_2.status == STARTED:
                if bool(answer == corrColor):
                    # keep track of stop time/frame for later
                    green_2.tStop = t  # not accounting for scr refresh
                    green_2.tStopRefresh = tThisFlipGlobal  # on global time
                    green_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'green_2.stopped')
                    # update status
                    green_2.status = FINISHED
                    green_2.setAutoDraw(False)
            
            # *RBGInfo_2* updates
            
            # if RBGInfo_2 is starting this frame...
            if RBGInfo_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                RBGInfo_2.frameNStart = frameN  # exact frame index
                RBGInfo_2.tStart = t  # local t and not account for scr refresh
                RBGInfo_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(RBGInfo_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'RBGInfo_2.started')
                # update status
                RBGInfo_2.status = STARTED
                RBGInfo_2.setAutoDraw(True)
            
            # if RBGInfo_2 is active this frame...
            if RBGInfo_2.status == STARTED:
                # update params
                pass
            
            # *R_2* updates
            
            # if R_2 is starting this frame...
            if R_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                R_2.frameNStart = frameN  # exact frame index
                R_2.tStart = t  # local t and not account for scr refresh
                R_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(R_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'R_2.started')
                # update status
                R_2.status = STARTED
                R_2.setAutoDraw(True)
            
            # if R_2 is active this frame...
            if R_2.status == STARTED:
                # update params
                pass
            
            # *G_2* updates
            
            # if G_2 is starting this frame...
            if G_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                G_2.frameNStart = frameN  # exact frame index
                G_2.tStart = t  # local t and not account for scr refresh
                G_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(G_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'G_2.started')
                # update status
                G_2.status = STARTED
                G_2.setAutoDraw(True)
            
            # if G_2 is active this frame...
            if G_2.status == STARTED:
                # update params
                pass
            
            # *B_2* updates
            
            # if B_2 is starting this frame...
            if B_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                B_2.frameNStart = frameN  # exact frame index
                B_2.tStart = t  # local t and not account for scr refresh
                B_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(B_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'B_2.started')
                # update status
                B_2.status = STARTED
                B_2.setAutoDraw(True)
            
            # if B_2 is active this frame...
            if B_2.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['b'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
            
            # *BaselineOverrideInfo* updates
            
            # if BaselineOverrideInfo is starting this frame...
            if BaselineOverrideInfo.status == NOT_STARTED and trial_counter >= 6:
                # keep track of start time/frame for later
                BaselineOverrideInfo.frameNStart = frameN  # exact frame index
                BaselineOverrideInfo.tStart = t  # local t and not account for scr refresh
                BaselineOverrideInfo.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(BaselineOverrideInfo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'BaselineOverrideInfo.started')
                # update status
                BaselineOverrideInfo.status = STARTED
                BaselineOverrideInfo.setAutoDraw(True)
            
            # if BaselineOverrideInfo is active this frame...
            if BaselineOverrideInfo.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=confirm,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                confirm.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if confirm.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in confirm.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "confirm" ---
        for thisComponent in confirm.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for confirm
        confirm.tStop = globalClock.getTime(format='float')
        confirm.tStopRefresh = tThisFlipGlobal
        thisExp.addData('confirm.stopped', confirm.tStop)
        trials.addData('Button3.numClicks', Button3.numClicks)
        if Button3.numClicks:
           trials.addData('Button3.timesOn', Button3.timesOn)
           trials.addData('Button3.timesOff', Button3.timesOff)
        else:
           trials.addData('Button3.timesOn', "")
           trials.addData('Button3.timesOff', "")
        trials.addData('blue_2.numClicks', blue_2.numClicks)
        if blue_2.numClicks:
           trials.addData('blue_2.timesOn', blue_2.timesOn)
           trials.addData('blue_2.timesOff', blue_2.timesOff)
        else:
           trials.addData('blue_2.timesOn', "")
           trials.addData('blue_2.timesOff', "")
        trials.addData('yellow_2.numClicks', yellow_2.numClicks)
        if yellow_2.numClicks:
           trials.addData('yellow_2.timesOn', yellow_2.timesOn)
           trials.addData('yellow_2.timesOff', yellow_2.timesOff)
        else:
           trials.addData('yellow_2.timesOn', "")
           trials.addData('yellow_2.timesOff', "")
        trials.addData('purple_2.numClicks', purple_2.numClicks)
        if purple_2.numClicks:
           trials.addData('purple_2.timesOn', purple_2.timesOn)
           trials.addData('purple_2.timesOff', purple_2.timesOff)
        else:
           trials.addData('purple_2.timesOn', "")
           trials.addData('purple_2.timesOff', "")
        trials.addData('cyan_2.numClicks', cyan_2.numClicks)
        if cyan_2.numClicks:
           trials.addData('cyan_2.timesOn', cyan_2.timesOn)
           trials.addData('cyan_2.timesOff', cyan_2.timesOff)
        else:
           trials.addData('cyan_2.timesOn', "")
           trials.addData('cyan_2.timesOff', "")
        trials.addData('red_2.numClicks', red_2.numClicks)
        if red_2.numClicks:
           trials.addData('red_2.timesOn', red_2.timesOn)
           trials.addData('red_2.timesOff', red_2.timesOff)
        else:
           trials.addData('red_2.timesOn', "")
           trials.addData('red_2.timesOff', "")
        trials.addData('green_2.numClicks', green_2.numClicks)
        if green_2.numClicks:
           trials.addData('green_2.timesOn', green_2.timesOn)
           trials.addData('green_2.timesOff', green_2.timesOff)
        else:
           trials.addData('green_2.timesOn', "")
           trials.addData('green_2.timesOff', "")
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        trials.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            trials.addData('key_resp.rt', key_resp.rt)
            trials.addData('key_resp.duration', key_resp.duration)
        # Run 'End Routine' code from code_4
        if key_resp == "p":
            trials.fished = True
        # the Routine "confirm" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "PauseEnd" ---
        # create an object to store info about Routine PauseEnd
        PauseEnd = data.Routine(
            name='PauseEnd',
            components=[text_11],
        )
        PauseEnd.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_5
        endTrialTime = globalClock.getTime(format='float')
        total_time = endTrialTime - startTrialTime
        
        # store start times for PauseEnd
        PauseEnd.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        PauseEnd.tStart = globalClock.getTime(format='float')
        PauseEnd.status = STARTED
        thisExp.addData('PauseEnd.started', PauseEnd.tStart)
        PauseEnd.maxDuration = None
        # skip Routine PauseEnd if its 'Skip if' condition is True
        PauseEnd.skipped = continueRoutine and not (BreakIt == True)
        continueRoutine = PauseEnd.skipped
        # keep track of which components have finished
        PauseEndComponents = PauseEnd.components
        for thisComponent in PauseEnd.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "PauseEnd" ---
        thisExp.currentRoutine = PauseEnd
        PauseEnd.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.35:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_11* updates
            
            # if text_11 is starting this frame...
            if text_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_11.frameNStart = frameN  # exact frame index
                text_11.tStart = t  # local t and not account for scr refresh
                text_11.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_11, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_11.started')
                # update status
                text_11.status = STARTED
                text_11.setAutoDraw(True)
            
            # if text_11 is active this frame...
            if text_11.status == STARTED:
                # update params
                pass
            
            # if text_11 is stopping this frame...
            if text_11.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_11.tStartRefresh + 0.35-frameTolerance:
                    # keep track of stop time/frame for later
                    text_11.tStop = t  # not accounting for scr refresh
                    text_11.tStopRefresh = tThisFlipGlobal  # on global time
                    text_11.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_11.stopped')
                    # update status
                    text_11.status = FINISHED
                    text_11.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=PauseEnd,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                PauseEnd.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if PauseEnd.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in PauseEnd.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "PauseEnd" ---
        for thisComponent in PauseEnd.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for PauseEnd
        PauseEnd.tStop = globalClock.getTime(format='float')
        PauseEnd.tStopRefresh = tThisFlipGlobal
        thisExp.addData('PauseEnd.stopped', PauseEnd.tStop)
        # Run 'End Routine' code from code_5
        if BreakIt == True:
            trials.finished = True
        thisExp.addData('colorList.routineEndVal', colorList)  # Save end Routine value
        
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if PauseEnd.maxDurationReached:
            routineTimer.addTime(-PauseEnd.maxDuration)
        elif PauseEnd.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.350000)
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 5.0 repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "congrats" ---
    # create an object to store info about Routine congrats
    congrats = data.Routine(
        name='congrats',
        components=[GoodJob],
    )
    congrats.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    GoodJob.setText('Good job!\n\nThe experiment will now close.\n\nThank you.')
    # store start times for congrats
    congrats.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    congrats.tStart = globalClock.getTime(format='float')
    congrats.status = STARTED
    thisExp.addData('congrats.started', congrats.tStart)
    congrats.maxDuration = 3
    # keep track of which components have finished
    congratsComponents = congrats.components
    for thisComponent in congrats.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "congrats" ---
    thisExp.currentRoutine = congrats
    congrats.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # is it time to end the Routine? (based on local clock)
        if tThisFlip > congrats.maxDuration-frameTolerance:
            congrats.maxDurationReached = True
            continueRoutine = False
        
        # *GoodJob* updates
        
        # if GoodJob is starting this frame...
        if GoodJob.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            GoodJob.frameNStart = frameN  # exact frame index
            GoodJob.tStart = t  # local t and not account for scr refresh
            GoodJob.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(GoodJob, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'GoodJob.started')
            # update status
            GoodJob.status = STARTED
            GoodJob.setAutoDraw(True)
        
        # if GoodJob is active this frame...
        if GoodJob.status == STARTED:
            # update params
            pass
        
        # if GoodJob is stopping this frame...
        if GoodJob.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > GoodJob.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                GoodJob.tStop = t  # not accounting for scr refresh
                GoodJob.tStopRefresh = tThisFlipGlobal  # on global time
                GoodJob.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'GoodJob.stopped')
                # update status
                GoodJob.status = FINISHED
                GoodJob.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=congrats,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            congrats.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if congrats.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in congrats.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "congrats" ---
    for thisComponent in congrats.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for congrats
    congrats.tStop = globalClock.getTime(format='float')
    congrats.tStopRefresh = tThisFlipGlobal
    thisExp.addData('congrats.stopped', congrats.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if congrats.maxDurationReached:
        routineTimer.addTime(-congrats.maxDuration)
    elif congrats.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    thisExp.addData('trial_counter.endExpVal', trial_counter)  # Save end experiment value
    
    
    
    
    
    
    
    
    
    # Run 'End Experiment' code from code_5
    thisExp.addData('corrColorVar.routineEndVal', corrColor_Var)  # Save end Routine value
    thisExp.addData('colorList.expEndVal', colorList)  # Save end Experiment value
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
