import contextlib
import io
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/lab03-recibos'))
from prepare_receipts import reference_for_split


class ReceiptReferencesTest(unittest.TestCase):
    def row(self,**changes):
        entities={'company':'SHOP','date':'30 DEC 17','address':'12 STREET','total':'102.40'}
        entities.update(changes)
        return {'key':'receipt','entities':entities}

    def test_complete_reference_keeps_original_values(self):
        row=self.row();excluded=[]
        self.assertEqual(reference_for_split(row,'train',excluded),row['entities'])
        self.assertEqual(excluded,[])

    def test_missing_training_labels_are_recorded_without_inventing_targets(self):
        for changes,field in [({'total':''},'total'),({'address':None},'address'),({'date':'  '},'date')]:
            with self.subTest(field=field),contextlib.redirect_stdout(io.StringIO()):
                row=self.row(**changes);excluded=[]
                self.assertIsNone(reference_for_split(row,'train',excluded))
                self.assertEqual(excluded,[{'source_key':'receipt','split':'train',
                    'missing_fields':[field],'entities':row['entities']}])

    def test_incomplete_test_reference_never_changes_test_denominator(self):
        excluded=[]
        with self.assertRaisesRegex(ValueError,'teste não será filtrado'):
            reference_for_split(self.row(total=''),'test',excluded)
        self.assertEqual(excluded,[])

    def test_wrong_label_type_is_not_silently_discarded(self):
        with self.assertRaisesRegex(ValueError,'Tipo inválido'):
            reference_for_split(self.row(total=102.4),'train',[])


if __name__=='__main__':unittest.main()
