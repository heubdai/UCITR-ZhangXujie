
from utils.LoadData import load_UEA

names = ["ArticularyWordRecognition","AtrialFibrillation","BasicMotions","CharacterTrajectories","Cricket","DuckDuckGeese","EigenWorms","Epilepsy","ERing","EthanolConcentration","FaceDetection","FingerMovements","HandMovementDirection","Handwriting","Heartbeat","JapaneseVowels","Libras","LSST","MotorImagery","NATOPS","PEMS-SF","PenDigits","PhonemeSpectra","RacketSports","SelfRegulationSCP1","SelfRegulationSCP2","SpokenArabicDigits","StandWalkJump","UWaveGestureLibrary","InsectWingbeat"]

for name in names:
    # train_X, train_y, test_X, test_y = load_UEA(name)
    print(name)
    # print(train_X.shape)
    # print(test_X.shape)