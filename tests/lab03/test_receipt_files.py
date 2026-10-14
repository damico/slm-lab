import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts/lab03-recibos'
sys.path.insert(0,str(SCRIPTS))
from receipt_common import preparation_lock,atomic_write


class ReceiptFilesTest(unittest.TestCase):
    def test_another_process_cannot_prepare_shared_images_until_lock_is_released(self):
        code="import sys;sys.path.insert(0,sys.argv[1]);from receipt_common import preparation_lock;\nwith preparation_lock(sys.argv[2]):print('acquired')"
        with tempfile.TemporaryDirectory() as folder:
            command=[sys.executable,'-c',code,str(SCRIPTS),folder]
            with preparation_lock(folder):
                result=subprocess.run(command,capture_output=True,text=True,timeout=5)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('Já existe uma preparação',result.stderr)
            result=subprocess.run(command,capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('acquired',result.stdout)

    def test_readers_see_old_complete_file_until_replacement_is_ready(self):
        import os
        original_replace=os.replace
        with tempfile.TemporaryDirectory() as folder:
            destination=Path(folder)/'receipt.img';destination.write_bytes(b'old complete image')
            def replace(source,target):
                self.assertEqual(destination.read_bytes(),b'old complete image')
                self.assertEqual(Path(source).read_bytes(),b'new complete image')
                original_replace(source,target)
            with patch('receipt_common.os.replace',side_effect=replace):atomic_write(destination,b'new complete image')
            self.assertEqual(destination.read_bytes(),b'new complete image')
            self.assertEqual(list(Path(folder).iterdir()),[destination])

    def test_failed_replacement_preserves_original_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as folder:
            destination=Path(folder)/'ocr.txt';destination.write_bytes(b'previous OCR')
            with patch('receipt_common.os.replace',side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):atomic_write(destination,b'new OCR')
            self.assertEqual(destination.read_bytes(),b'previous OCR')
            self.assertEqual(list(Path(folder).iterdir()),[destination])


if __name__=='__main__':unittest.main()
