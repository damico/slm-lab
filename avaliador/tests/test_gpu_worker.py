import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('gpu_worker',Path(__file__).resolve().parents[1]/'gpu_worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

class GpuWorkerTest(unittest.TestCase):
    def test_same_gpu_serializes_requests(self):
        entered=threading.Event();finished=threading.Event()
        def run():
            entered.set()
            with worker.gpu_slot('909'):finished.set()
        with patch.dict(sys.modules,{'torch':None}):
            with worker.gpu_slot('909'):
                thread=threading.Thread(target=run);thread.start()
                self.assertTrue(entered.wait(2));self.assertFalse(finished.wait(.1))
            thread.join(2);self.assertFalse(thread.is_alive());self.assertTrue(finished.is_set())
    def test_different_gpus_have_independent_slots(self):
        with patch.dict(sys.modules,{'torch':None}):
            with worker.gpu_slot('910'):
                with worker.gpu_slot('911'):pass
    def test_recorded_finqa_does_not_enter_gpu_queue(self):
        seen=[]
        method=worker.serialized(lambda self,payload:seen.append(payload) or 'recorded',skip_recorded=True)
        with patch.object(worker,'gpu_slot',side_effect=AssertionError('GPU queue should not run')):
            self.assertEqual(method(None,{'context':{},'mode':'recorded'}),'recorded')
        self.assertEqual(len(seen),1)

if __name__=='__main__':unittest.main()
