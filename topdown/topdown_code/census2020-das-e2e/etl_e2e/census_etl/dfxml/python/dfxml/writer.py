#!/usr/bin/env python3
#
# dfxml_gen.py: Generate DFXML
#
import sys
import os
import time
import os
import pwd
import sys
import datetime
import subprocess
import xml.etree.ElementTree as ET
import __main__
import atexit

__version__="0.1"

###
### Code for working with Apache Spark
### 

def json_to_xml(e,val):
    """Turns JSON in val into XML in e"""
    if type(val)==list:
        # A list is just a bunch of values
        l = ET.SubElement(e,'list')
        for v in val:
            json_to_xml(l,v)
        return
    if type(val)==dict:
        # A dict is a bunch of values, each with a key
        d = ET.SubElement(e,'dict')
        for (k,v) in val.items():
            json_to_xml(ET.SubElement(d,k), v)
        return
    if type(val) in [int,str,bool]:
        e.text = str(val)
        return
    raise RuntimeError("don't know how to build '{}'".format(val))


def git_commit():
    from subprocess import run,PIPE,SubprocessError
    try:
        s = run(['git','rev-parse','HEAD'],stdout=PIPE,encoding='utf-8')
        return s.stdout.strip()
    except SubprocessError as e:
        return ''

class DFXMLTimer:
    def __init__(self,dfxml,name):
        self.dfxml = dfxml
        self.name  = name
        self.t0    = 0

    def start(self):
        self.t0    = time.time()
        ET.SubElement(self.dfxml, 'timer', {'name':self.name, 'start':str(self.t0)})

    def stop(self):
        stop = time.time()
        ET.SubElement(self.dfxml, 'timer', {'name':self.name, 'start':str(self.t0), 'stop':str(stop), 'elapsed':str(stop-self.t0)})

class DFXMLWriter:
    def __init__(self,heartbeat=None,filename=None,prettyprint=False,logger=None):
        import time
        self.t0 = time.time()
        self.tlast = time.time()
        self.doc = ET.Element('dfxml')
        self.add_DFXML_creator(self.doc)
        self.filename = filename
        self.logger = logger
        self.timers = {}
        if self.filename:
            # Set up to automatically write to filename on exit...
            atexit.register(self.exiting,prettyprint=prettyprint)

    def timer(self,name):
        if name not in self.timers:
            self.timers[name] = DFXMLTimer(self.doc, name)
        return self.timers[name]

    def exiting(self,prettyprint=False):
        """Cleanup handling. Run done() and write the output file..."""
        if self.filename:
            self.done()
            self.writeToFilename(self.filename,prettyprint=prettyprint)
            self.filename = None

    def add_DFXML_creator(self,e):
        import __main__
        ee = ET.SubElement(e, 'creator', {'version':'1.0'})
        ET.SubElement(ee, 'program').text    = str(__main__.__file__)
        try:
            ET.SubElement(ee, 'version').text = str(__main__.__version__)
        except AttributeError as e:
            ET.SubElement(ee, 'version').text = ''
        ET.SubElement(ee, 'executable').text = sys.executable
        ET.SubElement(ee, 'git-commit').text = git_commit()
        self.add_DFXML_execution_environment(ee)
        
    def add_DFXML_execution_environment(self,e):
        ee = ET.SubElement(e, 'execution_environment')
        uname = os.uname()
        uname_fields = ['os_sysname','host','os_release','os_version','arch']
        for i in range(len(uname_fields)):
            ET.SubElement(ee, uname_fields[i]).text = uname[i]

        ET.SubElement(ee, 'command_line').text = " ".join([sys.executable] + sys.argv)
        ET.SubElement(ee, 'uid').text = str(os.getuid())
        ET.SubElement(ee, 'username').text = pwd.getpwuid(os.getuid())[0]
        ET.SubElement(ee, 'start_time').text = datetime.datetime.now().isoformat()
        try:
            import psutil
            ET.SubElement(ee, 'boot_time').text = datetime.datetime.fromtimestamp(psutil.boot_time()).isoformat()
        except ImportError:
            pass

    def timestamp(self,name):
        now = time.time()
        ET.SubElement(self.doc, 'timestamp', {'name':name,
                                                'delta':str(now - self.tlast),
                                                'total':str(now - self.t0)})
        if self.logger:
            self.logger("timestamp name:{}  delta:{:.4}  total:{}".
                        format(name,now-self.tlast,now-self.t0))
        self.tlast = now
        

    def add_rusage(self,node):
        import resource
        for who in ['RUSAGE_SELF','RUSAGE_CHILDREN']:
            ru = ET.SubElement(node, 'rusage', {'who':who})
            rusage = resource.getrusage(getattr(resource,who))
            rusage_fields = ['utime','stime','maxrss','ixrss','idrss','isrss',
                             'minflt','majflt','nswap','inblock','oublock',
                             'msgsnd','msgrcv','nsignals','nvcsw','nivcsw']

            for i in range(len(rusage_fields)):
                ET.SubElement(ru, rusage_fields[i]).text = str(rusage[i])
                if rusage_fields[i] in ['maxrss'] and self.logger:
                    self.logger("{}: {}".format(rusage_fields[i],rusage[i]))
            ET.SubElement(ru, 'pagesize').text = str(resource.getpagesize())


    def add_cpuinfo(self,node,interval=0.1):
        import psutil
        ET.SubElement(node, 'cpuinfo', {'interval':str(interval)}).text = str( psutil.cpu_percent(interval=interval,percpu=True) )

    def add_vminfo(self,node):
        import psutil
        vm = psutil.virtual_memory()
        ru = ET.SubElement(node, 'virtual_memory')
        for key in vm.__dir__():
            if key[0]!='_' and key not in ['index','count']:
                ET.SubElement(ru, key).text = str( getattr(vm, key))
        
    def comment(self,s):
        self.doc.insert(len(list(self.doc)), ET.Comment(s))
        if self.logger:
            self.logger(s)

    def asString(self):
        return ET.tostring(self.doc).decode('utf-8')

    def write(self,f,prettyprint=False):
        if prettyprint:
            f.write(self.prettyprint())
        else:
            f.write(self.asString())

    def writeToFilename(self,fname,prettyprint=False):
        self.write(open(fname,"w"),prettyprint=prettyprint)
            

    def prettyprint(self):
        import xml.dom.minidom
        return xml.dom.minidom.parseString( self.asString()).toprettyxml(indent='  ')

    def add_spark(self,node):
        """Connect to SPARK on local host and dump information. 
        Uses requests. Note: disables HTTPS certificate warnings."""
        import os
        import json
        from   urllib.request import urlopen
        import ssl
        if "SPARK_ENV_LOADED" not in os.environ:
            return        # no Spark

        spark = ET.SubElement(node, 'spark')
        try:
            import requests
            import urllib3
            urllib3.disable_warnings()
        except ImportError:
            ET.SubElement(spark,'error').text = "SPARK_ENV_LOADED present but requests module not available"
            return 

        host = 'localhost'
        p1 = 4040
        p2 = 4050
        import urllib.error
        for port in range(p1,p2+1):
            try:
                url = 'http://{}:{}/api/v1/applications/'.format(host,port)
                resp  = urlopen(url, context=ssl._create_unverified_context())
                spark_data = resp.read()
                break
            except (ConnectionError, ConnectionRefusedError, urllib.error.URLError) as e:
                continue
        if port>=p2:
            ET.SubElement(spark,'error').text = f"SPARK_ENV_LOADED present but no listener on {host} ports {p1}-{p2}"
            return

        # Looks like we have Spark!
        for app in json.loads(spark_data):
            app_id   = app['id']
            app_name = app['name']
            e = ET.SubElement(spark,'application',{'id':app_id,'name':app_name})

            attempt_count = 1
            for attempt in app['attempts']:
                e = ET.SubElement(spark,'attempt')
                json_to_xml(e,attempt)
            for param in ['jobs','allexecutors','storage/rdd']:
                url = f'http://{host}:{port}/api/v1/applications/{app_id}/{param}'
                resp = urlopen(url, context=ssl._create_unverified_context())
                data = resp.read()
                e = ET.SubElement(spark,param.replace("/","_"))
                json_to_xml(e,json.loads(data))


    def add_report(self,node,spark=True,rusage=True,vminfo=True):
        """Add the end of run report"""
        report = ET.SubElement(self.doc, 'report')
        if spark:
            self.add_spark(report)
        if rusage:
            self.add_rusage(report)
        try:
            import psutil
            ps = ET.SubElement(report, 'psutil')
            self.add_cpuinfo(ps)
            self.add_vminfo(ps)
        except ImportError:
            pass
        ET.SubElement(report, 'elapsed_seconds').text = str(time.time()-self.t0)

    def done(self):
        """Call when the program is finish"""
        self.add_report(self.doc)
        self.timestamp("done")

if __name__=="__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                                description="""Demo program. Run DFXML for this program and print the results. 
                                If you run it on a system with SPARK, you get the spark DFXML too!""") 
    arg_parser.add_argument("--write",help="Specify filename to write XML output to")
    arg_parser.add_argument("--debug",help="Print the output. Default unless --write is specified",action='store_true')
    args = arg_parser.parse_args()
    dfxml = DFXMLWriter()
    dfxml.timestamp("first")
    dfxml.timestamp("second")

    dfxml.timer("junky").start()
    time.sleep(.05)
    dfxml.timer("junky").stop()

    dfxml.timer("junky").start()
    time.sleep(.10)
    dfxml.timer("junky").stop()

    dfxml.timer("junky").start()
    time.sleep(.15)
    dfxml.timer("junky").stop()

    dfxml.timer("baby").start()
    time.sleep(.15)
    dfxml.timer("baby").stop()

    dfxml.done()
    dfxml.comment("Thanks")
    if args.write:
        dfxml.writeToFilename(args.write,prettyprint=True)
    if args.debug or not args.write:
        print(dfxml.prettyprint())

    # Demonstrate the failure system
    d2 = DFXMLWriter(filename="demo.dfxml",prettyprint=True)
    exit(0)
