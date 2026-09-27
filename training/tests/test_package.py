"""Tests also run against the installed wheel outside the source tree."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from l2s1_training.prepare_jev_data import prepare_custom, validate_dataset, convert
from l2s1_training.execution import execute_stage
from l2s1_training.common import save_status
from l2s1_training.jev_model_profiles import DEFAULT_PROFILES, read_profile


def row(name):
    return dict(id=name, workflow='test', state={'value':name},
        questions={'q':dict(type='noul', instructions='선택하세요')},
        gold={'q':dict(type='noul', label='true', probabilities={'false':.2, 'true':.8})})


class PackageTests(unittest.TestCase):
    def prepare(self, root):
        paths = []
        for split in ('train', 'development', 'test'):
            path = root/(split+'.jsonl')
            path.write_text(json.dumps(row(split), ensure_ascii=False)+'\n', encoding='utf-8')
            paths.append(path)
        prepare_custom(*paths, root/'data')
        return root/'data'

    def test_custom_state_can_be_plain_text(self):
        case = row('text')
        case['state'] = 'A plain text state, not JSON-encoded parquet.'
        self.assertEqual(convert(case, encoded=False)['request']['state'], case['state'])

    def test_installed_cli_and_bundled_registry(self):
        result = subprocess.run([sys.executable, '-m', 'l2s1_training', 'profiles'],
                                capture_output=True, text=True, check=True)
        self.assertEqual(len(json.loads(result.stdout)['models']), 5)
        for name in json.loads(DEFAULT_PROFILES.read_text())['models']:
            self.assertEqual(read_profile(name)['name'], name)

    def test_custom_prepare_rejects_overlap_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.prepare(root)
            self.assertEqual(validate_dataset(data)['splits']['train']['decisions'], 1)
            self.assertNotIn('gold', (data/'train-requests.jsonl').read_text())
            with self.assertRaises(FileExistsError):
                prepare_custom(root/'train.jsonl', root/'development.jsonl', root/'test.jsonl', data)
            with self.assertRaisesRegex(ValueError, 'leakage'):
                prepare_custom(root/'train.jsonl', root/'development.jsonl', root/'train.jsonl', root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_request_tampering_rejected_even_if_hash_is_updated(self):
        from l2s1_training.common import digest
        with tempfile.TemporaryDirectory() as tmp:
            data = self.prepare(Path(tmp))
            requests = data/'test-requests.jsonl'
            requests.write_text(json.dumps(dict(id='test', request={'state':'leaked label'}))+'\n')
            manifest = json.loads((data/'manifest.json').read_text())
            manifest['splits']['test']['requests_sha256'] = digest(requests)
            save_status(data/'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'payload'):
                validate_dataset(data)

    def test_unsupported_protocol_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.prepare(Path(tmp))
            manifest = json.loads((data/'manifest.json').read_text())
            manifest['protocol']['epochs'] = 4
            save_status(data/'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'protocol'):
                validate_dataset(data)

    def test_failed_child_and_missing_executable_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = dict(stages=[])
            persist = lambda: save_status(root/'run.json', status)
            for name, command in [('exit', [sys.executable, '-c', 'raise SystemExit(7)']),
                                  ('missing', [str(root/'missing')])]:
                with self.assertRaises((RuntimeError, FileNotFoundError)):
                    execute_stage(name, command, root, os.environ, status, persist, 10)
                record = json.loads((root/'run.json').read_text())['stages'][-1]
                self.assertEqual(record['status'], 'failed')
                self.assertGreater(record['elapsed_s'], 0)
            self.assertEqual(status['stages'][0]['exit_code'], 7)

    def test_timeout_kills_child_and_retains_live_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = dict(stages=[])
            persist = lambda: save_status(root/'run.json', status)
            with self.assertRaises(TimeoutError):
                execute_stage('sleep', [sys.executable, '-c', 'import time; time.sleep(60)'],
                              root, os.environ, status, persist, .2)
            saved = json.loads((root/'run.json').read_text())['stages'][0]
            self.assertEqual(saved['status'], 'timeout')
            self.assertIsNotNone(saved['exit_code'])
            self.assertLess(saved['elapsed_s'], 10)

    @unittest.skipIf(os.name == 'nt', 'POSIX SIGTERM process-group integration')
    def test_matrix_cancellation_persists_status_and_stops_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.prepare(root)
            profile = read_profile('gemma4')
            checkpoint = root/profile['revision']
            checkpoint.mkdir()
            (checkpoint/'config.json').write_text('{}')
            (root/profile['gguf']).write_bytes(b'fixture')
            exporter = root/'exporter'
            exporter.write_text('#!'+sys.executable+'\nimport time\ntime.sleep(60)\n')
            exporter.chmod(0o755)
            (root/'converter.py').write_text('')
            out = root/'run'
            command = [sys.executable, '-m', 'l2s1_training', 'run', '--models', 'gemma4', 'qwen3',
                '--checkpoint-root', str(root), '--data', str(data), '--gguf-root', str(root),
                '--exporter', str(exporter), '--evaluator', sys.executable,
                '--converter', str(root/'converter.py'), '--output', str(out)]
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic()+10
                while time.monotonic() < deadline:
                    if (out/'gemma4/export.log').exists():
                        break
                    if process.poll() is not None:
                        self.fail(process.communicate()[1].decode())
                    time.sleep(.02)
                else:
                    self.fail('export stage did not start')
                # Let the runner enter the child wait before sending SIGTERM.
                time.sleep(.05)
                process.send_signal(signal.SIGTERM)
                _, error = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 130, error.decode())
                runs = json.loads((out/'summary.json').read_text())['runs']
                self.assertEqual([r['status'] for r in runs], ['cancelled', 'not_started'])
                self.assertEqual(runs[0]['stages'][0]['status'], 'cancelled')
                self.assertIsNotNone(runs[0]['stages'][0]['exit_code'])
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def test_unsafe_profile_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = json.loads(DEFAULT_PROFILES.read_text())
            registry['models']['gemma4']['gguf'] = '../other.gguf'
            path = Path(tmp)/'models.json'
            path.write_text(json.dumps(registry))
            with self.assertRaisesRegex(ValueError, 'filename'):
                read_profile('gemma4', path)

    def test_all_models_failure_summary_is_durable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = subprocess.run([sys.executable, '-m', 'l2s1_training', 'run', '--models', 'all',
                '--checkpoint-root', str(root/'missing'), '--data', str(root/'data'),
                '--converter', str(root/'convert.py'), '--output', str(root/'run')], capture_output=True)
            self.assertEqual(run.returncode, 1)
            result = json.loads((root/'run/summary.json').read_text())
            self.assertEqual(len(result['runs']), 5)
            self.assertTrue(all(r['status']=='failed' for r in result['runs']))
            for r in result['runs']:
                self.assertEqual(json.loads((root/'run'/r['model']/'run.json').read_text()), r)


if __name__ == '__main__':
    unittest.main()
