/**
 * Security rules tests for the Realtime Database and Storage, run against the
 * local emulators:
 *
 *   npm run test:rules
 *
 * Each case writes what the app writes (or something it must not) and checks
 * the rules allow or deny it.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { assertFails, assertSucceeds, initializeTestEnvironment } from '@firebase/rules-unit-testing';
import { get, ref as dbRef, runTransaction, set, update } from 'firebase/database';
import { ref as storageRef, uploadBytes } from 'firebase/storage';

const ROOT = join(import.meta.dirname, '..');

async function main(): Promise<void> {
  const env = await initializeTestEnvironment({
    projectId: 'card-scanner-rules-test',
    database: { rules: readFileSync(join(ROOT, 'database.rules.json'), 'utf8'), host: '127.0.0.1', port: 9000 },
    storage: { rules: readFileSync(join(ROOT, 'storage.rules'), 'utf8'), host: '127.0.0.1', port: 9199 },
  });
  const alice = env.authenticatedContext('alice');
  const bob = env.authenticatedContext('bob');
  const anon = env.unauthenticatedContext();
  let failures = 0;
  const check = async (name: string, p: Promise<unknown>) => {
    try {
      await p;
      console.log(`ok    ${name}`);
    } catch (e) {
      failures++;
      console.log(`FAIL  ${name}: ${e instanceof Error ? e.message.split('\n')[0] : e}`);
    }
  };

  const card = { setCode: 'SVI', number: '001', total: '198', name: 'Pineco', count: 1, firstAdded: 1, lastAdded: 1 };
  const obs = { text: 'DLR 018/103', conf: 0.98, at: Date.now(), count: 1, total: '103', name: 'Growlithe', nameConf: 91, model: '2026-10-13T22:10:49.258671+00:00', image: 'scans/alice/DLR-018/1.jpg' };

  // Collection
  await check('owner writes own card', assertSucceeds(set(dbRef(alice.database(), 'users/alice/cards/SVI-001'), card)));
  await check('owner reads own collection', assertSucceeds(get(dbRef(alice.database(), 'users/alice/cards'))));
  await check('other user cannot read it', assertFails(get(dbRef(bob.database(), 'users/alice/cards'))));
  await check('other user cannot write it', assertFails(set(dbRef(bob.database(), 'users/alice/cards/SVI-001'), card)));
  await check('signed-out cannot read', assertFails(get(dbRef(anon.database(), 'users/alice'))));
  await check('bad card key rejected', assertFails(set(dbRef(alice.database(), 'users/alice/cards/svi-1'), card)));
  await check('unknown card field rejected', assertFails(set(dbRef(alice.database(), 'users/alice/cards/SVI-002'), { ...card, number: '002', extra: 1 })));

  // Observations
  await check('user writes own observation', assertSucceeds(set(dbRef(alice.database(), 'observations/DLR/018/alice'), obs)));
  await check('minimal observation (no name/image)', assertSucceeds(set(dbRef(alice.database(), 'observations/DLR/019/alice'), { text: 'DLR 019/103', conf: 0.9, at: 1, count: 1 })));
  await check(
    'user updates own observation in a transaction (as the app does)',
    assertSucceeds(runTransaction(dbRef(alice.database(), 'observations/DLR/018/alice'), (cur: { count?: number } | null) => ({ ...obs, count: (cur?.count ?? 0) + 1 }))),
  );
  await check('user reads own observation', assertSucceeds(get(dbRef(alice.database(), 'observations/DLR/018/alice'))));
  await check('cannot read another user\'s observation', assertFails(get(dbRef(bob.database(), 'observations/DLR/018/alice'))));
  await check('user marks own observation rejected', assertSucceeds(update(dbRef(alice.database(), 'observations/DLR/018/alice'), { rejected: true })));
  await check('cannot write as someone else', assertFails(set(dbRef(bob.database(), 'observations/DLR/018/alice'), obs)));
  await check('signed-out cannot write', assertFails(set(dbRef(anon.database(), 'observations/DLR/018/anon'), obs)));
  await check('nobody can read observations', assertFails(get(dbRef(alice.database(), 'observations/DLR'))));
  await check('bad set code rejected', assertFails(set(dbRef(alice.database(), 'observations/dlr/018/alice'), obs)));
  await check('confidence > 1 rejected', assertFails(set(dbRef(alice.database(), 'observations/DLR/020/alice'), { ...obs, conf: 3 })));
  await check('unknown observation field rejected', assertFails(set(dbRef(alice.database(), 'observations/DLR/021/alice'), { ...obs, foo: 'bar' })));

  // Storage
  const jpeg = new Uint8Array(200 * 1024);
  await check('upload own scan image', assertSucceeds(uploadBytes(storageRef(alice.storage(), 'scans/alice/DLR-018/1.jpg'), jpeg, { contentType: 'image/jpeg' })));
  await check('cannot upload into another user', assertFails(uploadBytes(storageRef(bob.storage(), 'scans/alice/DLR-018/2.jpg'), jpeg, { contentType: 'image/jpeg' })));
  await check('non-JPEG rejected', assertFails(uploadBytes(storageRef(alice.storage(), 'scans/alice/x/1.png'), jpeg, { contentType: 'image/png' })));
  await check('over 1 MB rejected', assertFails(uploadBytes(storageRef(alice.storage(), 'scans/alice/x/big.jpg'), new Uint8Array(1100 * 1024), { contentType: 'image/jpeg' })));
  await check('signed-out upload rejected', assertFails(uploadBytes(storageRef(anon.storage(), 'scans/anon/x/1.jpg'), jpeg, { contentType: 'image/jpeg' })));
  await check('outside scans/ rejected', assertFails(uploadBytes(storageRef(alice.storage(), 'other/1.jpg'), jpeg, { contentType: 'image/jpeg' })));

  await env.cleanup();
  console.log(failures ? `\n${failures} rule test(s) FAILED` : '\nall rule tests passed');
  process.exit(failures ? 1 : 0);
}

main().catch((e: unknown) => {
  console.error(e);
  process.exit(1);
});
