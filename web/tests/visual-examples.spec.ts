import { test, expect } from '@playwright/test';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { visibleDetections, outcome } from '../src/lib/vision/types';

test('bundled photos and all model observations match the frozen sample', async () => {
  const data = JSON.parse(readFileSync('static/trashnet/recorded.json', 'utf8'));
  expect(data.source.records).toHaveLength(120);
  for (const item of data.source.records) {
    expect(createHash('sha256').update(readFileSync('static' + item.image_url)).digest('hex')).toBe(item.sha256);
    for (const model of data.models) {
      const observation = model.observations[item.name];
      expect(observation.image_sha256).toBe(item.sha256);
      expect(observation.ground_truth).toBe(item.label);
    }
  }
  expect(data.models.map((model: { observations: Record<string, Parameters<typeof outcome>[0]> }) => Object.values(model.observations).filter((row) => outcome(row) === 'correct').length)).toEqual([91, 64, 0]);
});

test('normalized detection boxes are clipped and invalid coordinates are excluded', () => {
  const box = { xmin: -0.1, ymin: 0.2, xmax: 1.2, ymax: 0.9 };
  expect(visibleDetections([{ label: 'bottle', score: .9, box }, { label: 'bad', score: .95, box: { ...box, xmin: NaN } }, { label: 'reversed', score: .9, box: { ...box, xmin: .8, xmax: .4 } }, { label: 'low', score: .2, box }], .5)).toEqual([{ label: 'bottle', score: .9, box: { xmin: 0, ymin: .2, xmax: 1, ymax: .9 } }]);
});

test('gallery preserves abstention and changes image/model without mixing records', async ({ page }) => {
  await page.goto('/trashnet?lang=en');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('See the image');
  await expect(page.locator('.photo-card')).toHaveCount(120);
  await expect(page.locator('.stats')).toContainText('91/120');
  await expect(page.locator('.detail .decision')).toHaveText('Abstained');
  await page.getByLabel('Outcome', { exact: true }).selectOption('wrong');
  await expect(page.locator('.photo-card')).toHaveCount(24);
  await page.locator('.photo-card').first().click();
  await expect(page.locator('.detail .badge.wrong').first()).toBeVisible();
  await page.getByLabel('Model', { exact: true }).selectOption('smolvlm');
  await expect(page.locator('.photo-card')).toHaveCount(0);
  await expect(page.locator('.detail .decision')).toHaveText('Abstained');
  await expect(page.locator('.stats')).toContainText('0/120');
  await page.getByLabel('Outcome', { exact: true }).selectOption('abstained');
  await page.getByLabel('Ground-truth material', { exact: true }).selectOption('glass');
  await expect(page.locator('.photo-card')).toHaveCount(20);
});

test('recorded detector boxes, thresholds, sample links and empty results work', async ({ page }) => {
  await page.goto('/detect?lang=en');
  await expect(page.locator('.results')).toContainText('Recorded model inference');
  await expect(page.locator('.canvas > img')).toHaveAttribute('alt', 'glass330.jpg');
  await expect(page.locator('svg rect')).toHaveCount(1);
  await expect(page.locator('.object-row')).toContainText('bottle');
  const geometry = await page.locator('svg rect').evaluate((rect) => ({ x: Number(rect.getAttribute('x')), width: Number(rect.getAttribute('width')) }));
  expect(geometry.x).toBeGreaterThan(0); expect(geometry.width).toBeGreaterThan(10); expect(geometry.x + geometry.width).toBeLessThanOrEqual(100);
  await page.getByLabel('Show boxes').uncheck();
  await expect(page.locator('svg rect')).toHaveCount(0);
  await expect(page.locator('.object-row')).toHaveCount(1);
  await page.getByLabel('Show boxes').check();
  await page.getByRole('button', { name: 'metal91.jpg', exact: true }).click();
  await expect(page.locator('.results')).toContainText('No objects detected');
  await page.getByLabel('Minimum display score').fill('0.05');
  await page.goto('/detect?lang=en&sample=glass249.jpg');
  await expect(page.locator('.canvas > img')).toHaveAttribute('alt', 'glass249.jpg');
  await expect(page.locator('.results')).toContainText('No detection is recorded');
  await expect(page.locator('svg rect')).toHaveCount(0);
});

test('upload clears recorded results and rejects unsupported files', async ({ page }) => {
  await page.goto('/detect?lang=en');
  await expect(page.locator('svg rect')).toHaveCount(1);
  await page.locator('input[type=file]').setInputFiles('static/trashnet/images/plastic47.jpg');
  await expect(page.locator('.canvas > img')).toHaveAttribute('alt', 'plastic47.jpg');
  await expect(page.locator('svg rect')).toHaveCount(0);
  await expect(page.locator('.results')).not.toContainText('Recorded model inference');
  await page.locator('input[type=file]').setInputFiles({ name: 'bad.txt', mimeType: 'text/plain', buffer: Buffer.from('bad') });
  await expect(page.getByRole('alert')).toContainText('JPEG');
});

test('visual pages fit a mobile screen and support dark mode', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  for (const route of ['/trashnet', '/detect']) {
    await page.goto(route + '?lang=ko');
    await expect(page.locator('.visual-page .panel').first()).toBeVisible();
    await page.getByLabel('화면 모드').selectOption('dark');
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  }
});

test('real WASM detector finds a bottle and can run twice', async ({ page }) => {
  test.skip(!process.env.L2S1_TEST_DETECT_LIVE, 'Opt-in: downloads pinned public model and runs actual WASM inference');
  test.setTimeout(240_000);
  await page.goto('/detect?lang=en');
  await expect(page.locator('.results')).toContainText('Recorded model inference');
  await page.getByRole('button', { name: 'Run detection on this photo', exact: true }).click();
  await expect(page.locator('.results')).toContainText('Just run in this browser', { timeout: 180_000 });
  await expect(page.locator('.object-row')).toContainText('bottle');
  await expect(page.locator('svg rect')).toHaveCount(1);
  await page.getByRole('button', { name: 'plastic47.jpg', exact: true }).click();
  await page.getByRole('button', { name: 'Run detection on this photo', exact: true }).click();
  await expect(page.locator('.results')).toContainText('Just run in this browser', { timeout: 60_000 });
  await expect(page.locator('.object-row')).toContainText('bottle');
});
