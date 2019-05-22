import subprocess


run_scripts = [
    #e,f,b,t,es,i,a(augumentation),s(savelogs),n(normalize)
    # "python main.py -e 100 -f 64 -b 16 -t 0.85 -es 10 -i 7 -s -n",
    # "python main.py -e 100 -f 64 -b 16 -t 0.85 -es 10 -i 7 -s",
    "python main.py -e 100 -f 32 -b 16 -t 0.85 -es 10 -i 7 -s",
    # "python main.py -e 100 -f 32 -b 16 -t 0.85 -es 10 -i 7 -s",
    # "python main.py -e 100 -f 32 -b 16 -t 0.85 -es 10 -i 7 -s -n",
    "python main.py -e 100 -f 32 -b 16 -t 0.85 -es 10 -i 7 -s -n",


]









def main():
    for script in run_scripts:
        try:
            res = subprocess.check_output(script,shell=True)
        
        except Exception as e:
            print("例外args:", e.args)


if __name__ == "__main__":
    main()
