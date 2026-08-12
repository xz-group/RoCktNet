#!/usr/bin/env node

import { appendFile, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import {
  csvPath,
  downloadCsvLine,
  isPdfFile,
  launchIeeeContext,
  papersDir,
  readCsvPhysicalLine,
  resultsDir,
} from "./download_ieee_pdf.mjs";

const statusPath = path.join(resultsDir, "download_status.csv");

function integerArgument(name, defaultValue) {
  const index = process.argv.indexOf(name);
  if (index === -1) return defaultValue;
  const value = Number.parseInt(process.argv[index + 1], 10);
  if (!Number.isInteger(value)) throw new Error(`${name} must be an integer`);
  return value;
}

function hasFlag(name) {
  return process.argv.includes(name);
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

async function ensureStatusFile() {
  try {
    await stat(statusPath);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    await writeFile(
      statusPath,
      [
        "timestamp",
        "csv_line",
        "paper_id",
        "URL",
        "status",
        "file",
        "size_bytes",
        "method",
        "attempts",
        "error",
      ].join(",") + "\n",
      "utf8",
    );
  }
}

async function appendStatus(entry) {
  const fields = [
    entry.timestamp,
    entry.csvLine,
    entry.paperId,
    entry.url,
    entry.status,
    entry.file,
    entry.sizeBytes,
    entry.method,
    entry.attempts,
    entry.error,
  ];
  await appendFile(statusPath, fields.map(csvCell).join(",") + "\n", "utf8");
}

async function lastCsvPhysicalLine() {
  const text = (await readFile(csvPath, "utf8")).replace(/^\uFEFF/, "");
  const lines = text.split(/\r?\n/);
  for (let index = lines.length - 1; index >= 1; index -= 1) {
    if (lines[index].trim()) return index + 1;
  }
  throw new Error("The matched URL CSV has no data rows");
}

function randomDelay(minimum, maximum) {
  return Math.floor(minimum + Math.random() * (maximum - minimum + 1));
}

function looksLikeAuthenticationOrAccessFailure(error) {
  return /AUTH_REQUIRED|ACCESS_LIMITED|PDF control|sign.?in|log.?in|authentication|access|captcha|robot|denied|forbidden/i.test(
    error.message,
  );
}

async function closeExtraPages(context, preservedPages) {
  const preserved = new Set(preservedPages);
  for (const candidate of context.pages()) {
    if (!preserved.has(candidate) && !candidate.isClosed()) {
      await candidate.close().catch(() => {});
    }
  }
}

async function main() {
  const csvLastLine = await lastCsvPhysicalLine();
  const startLine = integerArgument("--start-line", 18);
  const endLine = integerArgument("--end-line", csvLastLine);
  const minDelayMs = integerArgument("--min-delay-ms", 8_000);
  const maxDelayMs = integerArgument("--max-delay-ms", 14_000);
  const maxAttempts = integerArgument("--max-attempts", 2);
  const noConfirm = hasFlag("--no-confirm");

  if (startLine < 2 || endLine < startLine || endLine > csvLastLine) {
    throw new Error(
      `Invalid range ${startLine}-${endLine}; CSV data occupies physical lines 2-${csvLastLine}`,
    );
  }
  if (minDelayMs < 0 || maxDelayMs < minDelayMs) {
    throw new Error("Invalid delay range");
  }
  if (maxAttempts < 1 || maxAttempts > 5) {
    throw new Error("--max-attempts must be between 1 and 5");
  }

  await ensureStatusFile();
  console.log(`Batch CSV range: ${startLine}-${endLine}`);
  console.log(`Output directory: ${papersDir}`);
  console.log(`Status log: ${statusPath}`);
  console.log(`Delay after downloads: ${minDelayMs}-${maxDelayMs} ms`);

  let stopRequested = false;
  process.on("SIGINT", () => {
    stopRequested = true;
    console.log("\nStop requested; finishing the current paper before closing...");
  });

  const context = await launchIeeeContext();
  const anchorPage = context.pages()[0] ?? (await context.newPage());
  await anchorPage.goto("about:blank").catch(() => {});
  const page = await context.newPage();
  await closeExtraPages(context, [anchorPage, page]);
  await page.bringToFront();
  const summary = {
    downloaded: 0,
    skippedExisting: 0,
    failed: 0,
  };
  let consecutiveFailures = 0;

  try {
    if (!noConfirm) {
      const firstRow = await readCsvPhysicalLine(startLine);
      await page.goto(firstRow.URL, {
        waitUntil: "domcontentloaded",
        timeout: 60_000,
      });
      const prompt = createInterface({ input, output });
      await prompt.question(
        "\nConfirm IEEE access in the Chrome window. Complete school authentication " +
          "if needed, then return here and press Enter to start the batch...",
      );
      prompt.close();
    }

    for (let csvLine = startLine; csvLine <= endLine; csvLine += 1) {
      if (stopRequested) break;

      const row = await readCsvPhysicalLine(csvLine);
      const targetPath = path.join(papersDir, `${csvLine}.pdf`);
      await closeExtraPages(context, [anchorPage, page]);
      await page.bringToFront().catch(() => {});
      console.log(
        `\n[${csvLine}/${endLine}] paper_id=${row.paper_id} ${row.URL}`,
      );

      if (await isPdfFile(targetPath)) {
        const existing = await stat(targetPath);
        summary.skippedExisting += 1;
        console.log(`SKIP: valid ${csvLine}.pdf already exists (${existing.size} bytes)`);
        await appendStatus({
          timestamp: new Date().toISOString(),
          csvLine,
          paperId: row.paper_id,
          url: row.URL,
          status: "skipped_existing",
          file: targetPath,
          sizeBytes: existing.size,
          method: "existing_valid_pdf",
          attempts: 0,
          error: "",
        });
        continue;
      }

      let result = null;
      let lastError = null;
      let attempts = 0;
      let authenticationPrompts = 0;
      for (attempts = 1; attempts <= maxAttempts; attempts += 1) {
        try {
          result = await downloadCsvLine(context, page, csvLine, row);
          break;
        } catch (error) {
          lastError = error;
          if (/AUTH_REQUIRED:/i.test(error.message) && authenticationPrompts < 2) {
            authenticationPrompts += 1;
            console.error(error.message);
            const prompt = createInterface({ input, output });
            await prompt.question(
              "\nComplete school authentication in the open Chrome tab. " +
                "When the IEEE PDF page has loaded, return here and press Enter to retry...",
            );
            prompt.close();
            await closeExtraPages(context, [anchorPage, page]);
            await page.bringToFront().catch(() => {});
            attempts -= 1;
            continue;
          }
          console.error(`Attempt ${attempts}/${maxAttempts} failed: ${error.message}`);
          await closeExtraPages(context, [anchorPage, page]);
          await page.bringToFront().catch(() => {});
          if (attempts < maxAttempts) {
            await page.waitForTimeout(3_000);
          }
        }
      }

      if (result) {
        consecutiveFailures = 0;
        summary.downloaded += 1;
        console.log(
          `SUCCESS: ${path.basename(result.targetPath)} (${result.size} bytes, ${result.method})`,
        );
        await appendStatus({
          timestamp: new Date().toISOString(),
          csvLine,
          paperId: row.paper_id,
          url: row.URL,
          status: result.status,
          file: result.targetPath,
          sizeBytes: result.size,
          method: result.method,
          attempts,
          error: "",
        });
      } else {
        consecutiveFailures += 1;
        summary.failed += 1;
        await appendStatus({
          timestamp: new Date().toISOString(),
          csvLine,
          paperId: row.paper_id,
          url: row.URL,
          status: "failed",
          file: targetPath,
          sizeBytes: "",
          method: "",
          attempts: maxAttempts,
          error: lastError?.message ?? "Unknown error",
        });

        if (lastError && looksLikeAuthenticationOrAccessFailure(lastError)) {
          console.error(
            "STOPPING: authentication, CAPTCHA, or access may require manual attention.",
          );
          break;
        }
        if (consecutiveFailures >= 3) {
          console.error(
            "STOPPING: three consecutive papers failed; manual review is required.",
          );
          break;
        }
      }

      if (csvLine < endLine && result) {
        const delay = randomDelay(minDelayMs, maxDelayMs);
        console.log(`Waiting ${(delay / 1000).toFixed(1)} seconds...`);
        await page.waitForTimeout(delay);
      }
    }
  } finally {
    await context.close();
  }

  console.log("\nBatch run finished.");
  console.log(`Downloaded: ${summary.downloaded}`);
  console.log(`Skipped existing: ${summary.skippedExisting}`);
  console.log(`Failed: ${summary.failed}`);
  console.log(`Status log: ${statusPath}`);
}

main().catch((error) => {
  console.error(`BATCH FAILED: ${error.stack ?? error.message}`);
  process.exitCode = 1;
});
