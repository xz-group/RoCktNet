#!/usr/bin/env node

import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import {
  mkdir,
  readFile,
  rename,
  stat,
  unlink,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { chromium } from "playwright-core";

const automationDir = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(automationDir, "..");
export const resultsDir = path.join(
  projectDir,
  "Data",
  "article_url_match_results",
);
export const csvPath = path.join(resultsDir, "matched_ids_and_urls.csv");
export const papersDir = path.join(resultsDir, "papers");
export const profileDir = path.join(resultsDir, "browser_profile");
export const chromePath = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";

function requestedLineNumber() {
  const index = process.argv.indexOf("--line");
  if (index === -1) return 18;
  const value = Number.parseInt(process.argv[index + 1], 10);
  if (!Number.isInteger(value) || value < 2) {
    throw new Error("--line must be a CSV physical line number of 2 or greater");
  }
  return value;
}

function parseTwoColumnCsvLine(line) {
  // This source file has exactly two columns. Handle quoted fields without
  // introducing a second CSV dependency.
  const values = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"') {
      if (quoted && line[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      values.push(value);
      value = "";
    } else {
      value += character;
    }
  }
  values.push(value);
  return values;
}

export async function readCsvPhysicalLine(lineNumber) {
  const text = (await readFile(csvPath, "utf8")).replace(/^\uFEFF/, "");
  const lines = text.split(/\r?\n/);
  if (lineNumber > lines.length || !lines[lineNumber - 1]) {
    throw new Error(`CSV physical line ${lineNumber} does not exist`);
  }

  const headers = parseTwoColumnCsvLine(lines[0]);
  const values = parseTwoColumnCsvLine(lines[lineNumber - 1]);
  const row = Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  if (!row.paper_id || !row.URL) {
    throw new Error(`CSV line ${lineNumber} is missing paper_id or URL`);
  }
  return row;
}

export async function isPdfFile(filePath) {
  try {
    const data = await readFile(filePath);
    return data.length > 5 && data.subarray(0, 5).toString("ascii") === "%PDF-";
  } catch {
    return false;
  }
}

async function savePdfBuffer(buffer, targetPath) {
  if (buffer.length <= 5 || buffer.subarray(0, 5).toString("ascii") !== "%PDF-") {
    return false;
  }
  const partialPath = `${targetPath}.part`;
  await writeFile(partialPath, buffer);
  try {
    await unlink(targetPath);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  await rename(partialPath, targetPath);
  return true;
}

async function saveBrowserDownload(download, targetPath) {
  const partialPath = `${targetPath}.part`;
  await download.saveAs(partialPath);
  if (!(await isPdfFile(partialPath))) {
    await unlink(partialPath).catch(() => {});
    throw new Error(`Browser download was not a valid PDF: ${download.suggestedFilename()}`);
  }
  try {
    await unlink(targetPath);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  await rename(partialPath, targetPath);
}

async function firstVisiblePdfControl(page) {
  const candidates = [
    page.getByRole("link", { name: /^\s*PDF\s*$/i }),
    page.getByRole("button", { name: /^\s*PDF\s*$/i }),
    page.getByRole("link", { name: /PDF/i }),
    page.getByRole("button", { name: /PDF/i }),
    page.locator('a[href*="/stamp/"], a[href*="getPDF"], a[href$=".pdf"]').first(),
  ];

  for (const candidate of candidates) {
    if ((await candidate.count()) > 0 && (await candidate.first().isVisible())) {
      return candidate.first();
    }
  }
  return null;
}

async function clickAndObserve(sourcePage, control) {
  const oldUrl = sourcePage.url();
  const downloadPromise = sourcePage
    .waitForEvent("download", { timeout: 20_000 })
    .then((download) => ({ download, page: sourcePage }))
    .catch(() => new Promise(() => {}));
  const newPagePromise = sourcePage
    .context()
    .waitForEvent("page", { timeout: 20_000 })
    .then(async (page) => {
      await page.waitForLoadState("domcontentloaded", { timeout: 30_000 }).catch(() => {});
      return { download: null, page };
    })
    .catch(() => new Promise(() => {}));
  const navigationPromise = sourcePage
    .waitForURL((url) => url.toString() !== oldUrl, { timeout: 20_000 })
    .then(() => ({ download: null, page: sourcePage }))
    .catch(() => new Promise(() => {}));

  await control.click();
  return Promise.race([
    downloadPromise,
    newPagePromise,
    navigationPromise,
    sourcePage.waitForTimeout(20_000).then(() => ({
      download: null,
      page: sourcePage,
    })),
  ]);
}

async function tryViewerDownload(page, targetPath) {
  const selectors = [
    "#download",
    'cr-icon-button[aria-label*="Download" i]',
    'button[aria-label*="Download" i]',
    'a[aria-label*="Download" i]',
    'button[title*="Download" i]',
    'a[title*="Download" i]',
  ];

  for (const selector of selectors) {
    const button = page.locator(selector).first();
    if ((await button.count()) === 0 || !(await button.isVisible().catch(() => false))) {
      continue;
    }
    const downloadPromise = page
      .waitForEvent("download", { timeout: 20_000 })
      .catch(() => null);
    await button.click().catch(() => {});
    const download = await downloadPromise;
    if (download) {
      await saveBrowserDownload(download, targetPath);
      return true;
    }
  }
  return false;
}

function extractCandidateUrls(html, baseUrl) {
  const urls = [];
  const pattern = /(?:src|href)=["']([^"']+)["']/gi;
  for (const match of html.matchAll(pattern)) {
    if (!/pdf|stamp|getPDF/i.test(match[1])) continue;
    try {
      urls.push(new URL(match[1], baseUrl).toString());
    } catch {
      // Ignore malformed links.
    }
  }
  return urls;
}

async function fetchPdfThroughAuthenticatedContext(context, initialUrls, targetPath, referer) {
  const queue = [...initialUrls];
  const visited = new Set();

  while (queue.length > 0 && visited.size < 15) {
    const candidate = queue.shift();
    if (!candidate || visited.has(candidate) || !/^https?:/i.test(candidate)) continue;
    visited.add(candidate);
    console.log(`Trying authenticated PDF URL: ${candidate}`);

    const response = await context.request
      .get(candidate, {
        headers: { referer },
        timeout: 45_000,
        failOnStatusCode: false,
      })
      .catch(() => null);
    if (!response || !response.ok()) continue;

    const body = await response.body();
    if (await savePdfBuffer(body, targetPath)) return true;

    const contentType = response.headers()["content-type"] ?? "";
    if (/html|text/i.test(contentType)) {
      queue.push(...extractCandidateUrls(body.toString("utf8"), candidate));
    }
  }
  return false;
}

async function collectPdfCandidates(page) {
  const urls = [page.url(), ...page.frames().map((frame) => frame.url())];
  const linkUrls = await page
    .locator("a[href], iframe[src], embed[src]")
    .evaluateAll((elements) =>
      elements
        .map((element) => element.href || element.src)
        .filter(Boolean),
    )
    .catch(() => []);
  urls.push(...linkUrls);
  return [...new Set(urls.filter((url) => /pdf|stamp|getPDF/i.test(url)))];
}

function isAuthenticationUrl(url) {
  return /(?:login|signin|sso|saml|shibboleth|\/idp\/profile\/)/i.test(url);
}

export async function launchIeeeContext() {
  await mkdir(papersDir, { recursive: true });
  await mkdir(profileDir, { recursive: true });
  return chromium.launchPersistentContext(profileDir, {
    executablePath: chromePath,
    headless: false,
    acceptDownloads: true,
    downloadsPath: papersDir,
    viewport: null,
    args: ["--start-maximized"],
  });
}

export async function downloadCsvLine(
  context,
  page,
  lineNumber,
  row,
  { navigate = true } = {},
) {
  const targetPath = path.join(papersDir, `${lineNumber}.pdf`);
  if (await isPdfFile(targetPath)) {
    const existing = await stat(targetPath);
    return {
      status: "skipped_existing",
      targetPath,
      size: existing.size,
      method: "existing_valid_pdf",
    };
  }

  if (navigate) {
    await page.goto(row.URL, { waitUntil: "domcontentloaded", timeout: 60_000 });
  }
  await page.bringToFront();

  const pdfControl = await firstVisiblePdfControl(page);
  if (!pdfControl) {
    throw new Error(
      "Could not find a visible PDF control. Confirm the article page is open and access is granted.",
    );
  }

  console.log("Clicking the article PDF control...");
  const observed = await clickAndObserve(page, pdfControl);
  let method = "browser_download";
  if (observed.download) {
    await saveBrowserDownload(observed.download, targetPath);
  } else {
    const pdfPage = observed.page;
    let keepPdfPageOpen = false;
    try {
      await pdfPage.waitForTimeout(2_000);
      console.log(`PDF page: ${pdfPage.url()}`);
      if (/[?&]denied(?:=|&|$)/i.test(pdfPage.url())) {
        throw new Error(
          `ACCESS_LIMITED: IEEE denied PDF access at ${pdfPage.url()}`,
        );
      }
      if (isAuthenticationUrl(pdfPage.url())) {
        keepPdfPageOpen = true;
        throw new Error(
          `AUTH_REQUIRED: complete school authentication in the open browser tab (${pdfPage.url()})`,
        );
      }

      const clickedViewerDownload = /(?:stamp|pdf)/i.test(pdfPage.url())
        ? await tryViewerDownload(pdfPage, targetPath)
        : false;
      if (clickedViewerDownload) {
        method = "viewer_download";
      } else {
        const candidateUrls = await collectPdfCandidates(pdfPage);
        const fetched = await fetchPdfThroughAuthenticatedContext(
          context,
          candidateUrls,
          targetPath,
          row.URL,
        );
        if (!fetched) {
          throw new Error(
            "The PDF page opened, but neither the viewer download button nor an authenticated PDF URL produced a valid PDF.",
          );
        }
        method = "authenticated_pdf_request";
      }
    } finally {
      if (!keepPdfPageOpen && pdfPage !== page && !pdfPage.isClosed()) {
        await pdfPage.close().catch(() => {});
      }
    }
  }

  if (!(await isPdfFile(targetPath))) {
    throw new Error("The saved file failed the %PDF header validation");
  }
  const saved = await stat(targetPath);
  return { status: "downloaded", targetPath, size: saved.size, method };
}

async function main() {
  const lineNumber = requestedLineNumber();
  const row = await readCsvPhysicalLine(lineNumber);
  const targetPath = path.join(papersDir, `${lineNumber}.pdf`);

  console.log(`CSV line: ${lineNumber}`);
  console.log(`paper_id: ${row.paper_id}`);
  console.log(`URL: ${row.URL}`);
  console.log(`Target: ${targetPath}`);

  if (await isPdfFile(targetPath)) {
    const existing = await stat(targetPath);
    console.log(`A valid PDF already exists (${existing.size} bytes); nothing to do.`);
    return;
  }

  const context = await launchIeeeContext();

  try {
    const page = context.pages()[0] ?? (await context.newPage());
    await page.goto(row.URL, { waitUntil: "domcontentloaded", timeout: 60_000 });
    console.log(`Opened: ${page.url()}`);

    const prompt = createInterface({ input, output });
    await prompt.question(
      "\nComplete school authentication in the Chrome window if needed. " +
        "When the IEEE article page is ready, return here and press Enter...",
    );
    prompt.close();

    const result = await downloadCsvLine(context, page, lineNumber, row, {
      navigate: false,
    });
    console.log(`SUCCESS: ${result.targetPath}`);
    console.log(`PDF size: ${result.size} bytes`);
    console.log(`Method: ${result.method}`);
  } finally {
    await context.close();
  }
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  main().catch((error) => {
    console.error(`FAILED: ${error.stack ?? error.message}`);
    process.exitCode = 1;
  });
}
