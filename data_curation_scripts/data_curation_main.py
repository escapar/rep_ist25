import cs_designite_runner
import cs_code_split_runner
import cs_learning_data_generator
import tokenizer_runner
import java_designite_runner
import java_codeSplit_runner
import java_learning_data_generator
DATA_BASE_PATH = 'D:\\JSS_DeepLearemingCS\\DeepLearningSmells-master\\smellDetection\\data'
CS_REPO_SOURCE_FOLDER = DATA_BASE_PATH + '\\all_cs_repos'
BATCH_FILES_FOLDER = DATA_BASE_PATH + '\\BatchFiles'
CS_SMELLS_RESULTS_FOLDER = DATA_BASE_PATH + '\\designite_out'
CS_DESIGNITE_CONSOLE_PATH = 'C:\\Program Files (x86)\\Designite\\DesigniteConsole.exe'
CS_CODE_SPLIT_OUT_FOLDER_CLASS = DATA_BASE_PATH + '\\codesplit_out_class'
CS_CODE_SPLIT_OUT_FOLDER_METHOD = DATA_BASE_PATH + '\\codesplit_out_method'
CS_CODE_SPLIT_MODE_CLASS = '-c'
CS_CODE_SPLIT_MODE_METHOD = '-m'
CS_CODE_SPLIT_EXE_PATH = 'D:\\Dev\\codeSplit\\CodeSplit\\bin\\Release\\CodeSplit.exe'
CS_LEARNING_DATA_FOLDER_BASE = 'D:\\JSS_DeepLearemingCS\\cs_code_cpu3\\cs_code'
TOKENIZER_EXE_PATH = 'D:\\JSS_DeepLearemingCS\\tokenizer-master\\src\\tokenizer.exe'
CS_TOKENIZER_OUT_PATH = 'D:\\JSS_DeepLearemingCS\\cs_code_cpu3\\cs_token_cpu3'
JAVA_REPO_SOURCE_FOLDER = DATA_BASE_PATH + '\\all_java_repos'
JAVA_SMELLS_RESULTS_FOLDER = DATA_BASE_PATH + '\\designite_out_java'
DESIGNITE_JAVA_JAR_PATH = 'D:\\research\\smellDetectionML\\dj\\DesigniteJava.jar'
JAVA_CODE_SPLIT_OUT_FOLDER_CLASS = DATA_BASE_PATH + '\\codesplit_java_class'
JAVA_CODE_SPLIT_OUT_FOLDER_METHOD = DATA_BASE_PATH + '\\codesplit_java_method'
JAVA_CODE_SPLIT_MODE_CLASS = 'class'
JAVA_CODE_SPLIT_MODE_METHOD = 'method'
JAVA_CODE_SPLIT_EXE_PATH = 'D:\\research\\smellDetectionML\\CodeSplitJava\\target\\CodeSplitJava.jar'
JAVA_LEARNING_DATA_FOLDER_BASE = DATA_BASE_PATH + '\\smellML_data_java'
JAVA_TOKENIZER_OUT_PATH = DATA_BASE_PATH + '\\tokenizer_out_java'
if __name__ == '__main__':
    tokenizer_runner.tokenize('CSharp', CS_LEARNING_DATA_FOLDER_BASE, CS_TOKENIZER_OUT_PATH, TOKENIZER_EXE_PATH)