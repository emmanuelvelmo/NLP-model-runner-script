from cx_Freeze import setup, Executable

setup(name="NLP model runner script", executables=[Executable("NLP model runner script.py")], options={"build_exe": {"excludes": ["tkinter"]}})