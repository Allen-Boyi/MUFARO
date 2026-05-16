import 'dotenv/config';

import express    from 'express';
import bodyParser from 'body-parser';
import path       from 'path';
import fs         from 'fs-extra';
import { fileURLToPath } from 'url';
import pino       from 'pino';
import QRCode     from 'qrcode';
import Groq       from 'groq-sdk';
import {
    makeWASocket,
    useMultiFileAuthState,
    makeCacheableSignalKeyStore,
    fetchLatestBaileysVersion,
    DisconnectReason,
    delay,
    normalizeMessageContent,
} from '@whiskeysockets/baileys';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// ──────────────────────────────────────────────
// Groq Client
// ──────────────────────────────────────────────
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// ──────────────────────────────────────────────
// System Instructions (Mufaro Persona)
// ──────────────────────────────────────────────
const SYSTEM_INSTRUCTIONS = `
Persona:
You are Mufaro, a warm, encouraging, and expert educational assistant for primary and secondary school students. You are patient, never condescending, and always make the student feel capable of understanding. You never do the student's homework for them — instead, you guide them to the answer.


Step 1 — Analyze (Detect Level)
Before responding, silently assess the student's academic level based on their vocabulary, sentence structure, and question complexity.

Primary (ages 6–12): Use short sentences, everyday words, and physical/tangible examples. Avoid jargon entirely. Keep explanations to 3–5 sentences max per point.
Secondary (ages 13–18): Introduce subject-specific terminology (define it on first use). Allow for more abstract reasoning and multi-step explanations.

If the level is unclear, default to Primary and adjust upward if the student responds with sophistication.

Step 2 — Explain (Core Teaching)
Provide a direct, accurate, and well-structured explanation of the concept.
- Use clear headings to organize the response.
- Bold every key term the first time it appears.
- Match sentence complexity strictly to the detected level.
- If the concept has multiple parts, break them into numbered steps or a short list.

Step 3 — Connect (Real-World Example)
Provide one specific, relatable real-world example that illustrates the concept. The example must:
- Be age-appropriate and culturally accessible.
- Directly map to the explanation (no loose analogies).
- Be introduced with a phrase like "Think of it this way…" or "Here's an example from everyday life…" (translated into the student's language).

Step 4 — Verify (Check Understanding)
End every response with one question that requires the student to apply what they just learned — not just recall it.
- For Primary: Make it concrete and imaginative ("What do you think would happen if…?")
- For Secondary: Make it analytical or evaluative ("How would this change if…?" or "Can you think of a case where this might not apply?")
- Frame it warmly — never make it feel like a test.
- Write this question in the student's detected language.

Step 5 — Correct with Care (Misconception Handling)
If the student gives a wrong or partially correct answer:
- Never say "wrong" or "incorrect."
- Acknowledge what they got right first.
- Gently redirect: "You're on the right track — let's look at this part more closely…" (translated into the student's language).
- Re-explain only the specific misunderstood part, not the whole concept.

Boundaries:
- Only assist with academic subjects (Math, Science, Languages, History, Geography, etc.).
- If asked to write an essay, solve an exam, or complete homework for the student, decline kindly and offer to guide them through it step by step instead. Respond to this boundary message in the student's language.
- If a question is off-topic or inappropriate, redirect warmly: "That's outside what I can help with, but let's get back to your studies!" (in the student's language).
`;

// ──────────────────────────────────────────────
// Conversation History Config
// ──────────────────────────────────────────────
const MAX_HISTORY      = 10;              // Max messages kept per student (5 exchanges)
const TTL_MS           = 24 * 60 * 60 * 1000; // 24 hours
const CLEANUP_INTERVAL = 60 * 60 * 1000;      // Run cleanup every 1 hour

/**
 * In-memory conversation store
 * {
 *   "student_id": {
 *     messages:   [ {role, content}, ... ],
 *     lastActive: <Date>,
 *     createdAt:  <Date>
 *   }
 * }
 */
const conversationHistories = new Map();

// ── Background cleanup (every hour) ──────────
setInterval(() => {
    const now     = Date.now();
    let   removed = 0;
    for (const [id, data] of conversationHistories) {
        if (now - data.createdAt.getTime() >= TTL_MS) {
            conversationHistories.delete(id);
            removed++;
            console.log(`[Cleanup] Session deleted for student: ${id}`);
        }
    }
    if (removed > 0) {
        console.log(`[Cleanup] Removed ${removed} expired session(s).`);
    } else {
        console.log('[Cleanup] No expired sessions found.');
    }
}, CLEANUP_INTERVAL);

// ──────────────────────────────────────────────
// Core Mufaro Function (calls Groq directly)
// ──────────────────────────────────────────────
async function askMufaro(studentId, userMessage) {
    const now = new Date();

    // Create new session if needed
    if (!conversationHistories.has(studentId)) {
        conversationHistories.set(studentId, {
            messages:   [],
            lastActive: now,
            createdAt:  now,
        });
    }

    const session = conversationHistories.get(studentId);

    // Force-reset if session exceeded 24 hours
    if (now.getTime() - session.createdAt.getTime() >= TTL_MS) {
        conversationHistories.set(studentId, {
            messages:   [],
            lastActive: now,
            createdAt:  now,
        });
        console.log(`[Session] 24 hr TTL reached — reset session for: ${studentId}`);
    }

    session.lastActive = now;
    session.messages.push({ role: 'user', content: userMessage });

    // Keep only the last MAX_HISTORY messages
    const trimmedHistory = session.messages.slice(-MAX_HISTORY);

    const messages = [
        { role: 'system', content: SYSTEM_INSTRUCTIONS },
        ...trimmedHistory,
    ];

    // Call Groq
    const completion = await groq.chat.completions.create({
        model:              'llama-3.3-70b-versatile',
        messages,
        temperature:        0.7,
        max_completion_tokens: 1024,
    });

    const reply = completion.choices[0].message.content;
    session.messages.push({ role: 'assistant', content: reply });

    return reply;
}

// ──────────────────────────────────────────────
// Server Config
// ──────────────────────────────────────────────
const PORT            = Number(process.env.PORT) || 7860;
const BOT_SESSION_DIR = process.env.BOT_SESSION_DIR || path.join(__dirname, 'bot_session');

// ──────────────────────────────────────────────
// Session Config
// ──────────────────────────────────────────────
const STOP_WORDS         = ['stop', 'bye', 'exit', 'quit'];
const INACTIVITY_TIMEOUT = 5 * 60 * 1000; // 5 minutes

// ──────────────────────────────────────────────
// Active WhatsApp Bot Sessions
// { "jid" → { timer, activatedAt } }
// ──────────────────────────────────────────────
const activeSessions = new Map();

// ──────────────────────────────────────────────
// Messages
// ──────────────────────────────────────────────

/** Returns "Morning" | "Afternoon" | "Evening" | "Night" based on current hour */
function getTimePeriod() {
    const hour = new Date().getHours();
    if (hour >= 5  && hour < 12) return 'Morning';
    if (hour >= 12 && hour < 17) return 'Afternoon';
    if (hour >= 17 && hour < 21) return 'Evening';
    return 'Night';
}

/** Static fallback — used when Groq is unreachable */
function getFallbackGreeting(name) {
    const firstName  = name ? name.split(' ')[0] : 'there';
    const timePeriod = getTimePeriod();
    return `👋 Good ${timePeriod}, ${firstName}! How are you doing?\n\nI'm *Mufaro*, your educational assistant 😊\n\nWhat do you want to learn today? What's on your mind? Go ahead, I'm all yours!\n\n_Type *stop* or *bye* at any time to end our session._`;
}

/** Generates a warm, time-aware greeting via Groq. Falls back to static on failure. */
async function generateGreeting(name) {
    const firstName  = name ? name.split(' ')[0] : 'there';
    const timePeriod = getTimePeriod();

    try {
        const completion = await groq.chat.completions.create({
            model:   'llama-3.3-70b-versatile',
            messages: [
                {
                    role:    'system',
                    content: `You are Mufaro, a warm and encouraging educational assistant for school students on WhatsApp.
Your job is to write a short, friendly opening greeting message.

Rules:
- Start with "Good ${timePeriod}, ${firstName}!" — use exactly this time-of-day phrase and the student's name.
- Follow with 2–3 warm sentences introducing yourself as Mufaro, their educational assistant.
- Invite them to share what they want to learn today.
- End with: "_Type *stop* or *bye* at any time to end our session._"
- Use WhatsApp-friendly formatting: *bold* for your name, _italic_ for the footer line.
- Keep it under 60 words total. No headers. No bullet points. Plain WhatsApp message.
- Output ONLY the greeting message — nothing else, no commentary.`,
                },
                {
                    role:    'user',
                    content: `Generate the greeting now for a student named ${firstName} at ${timePeriod.toLowerCase()} time.`,
                },
            ],
            temperature:           0.85,
            max_completion_tokens: 150,
        });

        const generated = completion.choices[0].message.content?.trim();
        if (generated && generated.length > 10) {
            console.log(`[Greeting] Groq-generated for ${firstName} (${timePeriod})`);
            return generated;
        }
        throw new Error('Empty response from Groq');
    } catch (err) {
        console.warn(`[Greeting] Groq failed — using fallback. Reason: ${err.message}`);
        return getFallbackGreeting(name);
    }
}

const GOODBYE_MESSAGE    = `📚 Your session with Mufaro has ended. Well done for studying today!\n\n_Message me whenever you're ready to learn again. Good luck! 🌟_`;
const INACTIVITY_GOODBYE = `⏰ Your Mufaro session closed due to 5 minutes of inactivity.\n\n_Send me a message whenever you want to continue learning! 📖_`;

// ──────────────────────────────────────────────
// Express App
// ──────────────────────────────────────────────
import('events').then((events) => {
    events.EventEmitter.defaultMaxListeners = 500;
});

let botQrDataUrl = null;

const app = express();
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static(__dirname));

// ── Health check ──────────────────────────────
app.get('/health', (_req, res) => {
    res.json({
        ok:             true,
        service:        'mufaro-bot',
        botQrReady:     Boolean(botQrDataUrl),
        activeSessions: activeSessions.size,
    });
});

// ── QR code page ─────────────────────────────
app.get('/bot-qr', (_req, res) => {
    if (!botQrDataUrl) {
        return res.status(404).json({ error: 'No active QR. Bot may already be linked or still starting.' });
    }
    res.type('html').send(`<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Mufaro Bot — scan to link</title></head>
<body style="font-family:system-ui,sans-serif;max-width:24rem;margin:2rem auto;padding:0 1rem;text-align:center">
  <h1 style="font-size:1.15rem">Link <strong>Mufaro Bot</strong></h1>
  <img src="${botQrDataUrl}" alt="WhatsApp QR" width="280" height="280"
       style="margin:1rem auto;border-radius:12px;box-shadow:0 4px 24px rgba(0,0,0,.08)"/>
  <p style="font-size:0.8rem;color:#666">
    WhatsApp → Linked devices → Link a device → scan this code.
  </p>
</body></html>`);
});

// ── REST chat endpoint (mirrors app.py /chat) ─
app.post('/chat', async (req, res) => {
    const { student_id, message } = req.body;
    if (!student_id || !message) {
        return res.status(400).json({ error: 'student_id and message are required.' });
    }

    for (let attempt = 0; attempt < 3; attempt++) {
        try {
            const reply = await askMufaro(student_id, message);
            return res.json({ message: reply });
        } catch (err) {
            if (String(err).includes('429')) {
                if (attempt < 2) {
                    await new Promise((r) => setTimeout(r, 2000));
                    continue;
                }
                return res.json({ message: "Mufaro is helping other students right now. Please try again in a moment! 😊" });
            }
            return res.status(500).json({ error: String(err) });
        }
    }
});

// ── Clear one student's history ───────────────
app.delete('/chat/:student_id', (req, res) => {
    const { student_id } = req.params;
    if (conversationHistories.has(student_id)) {
        conversationHistories.delete(student_id);
        return res.json({ message: `History cleared for student: ${student_id}` });
    }
    res.json({ message: 'No history found for this student.' });
});

// ── Admin: clear ALL histories ────────────────
app.delete('/admin/clear-all', (_req, res) => {
    const count = conversationHistories.size;
    conversationHistories.clear();
    res.json({ message: `All sessions cleared. ${count} student(s) removed.` });
});

// ── Admin: list active sessions ───────────────
app.get('/admin/sessions', (_req, res) => {
    const now      = Date.now();
    const sessions = [];
    for (const [studentId, data] of conversationHistories) {
        const ageMs     = now - data.createdAt.getTime();
        const idleMs    = now - data.lastActive.getTime();
        const expiresMs = Math.max(0, TTL_MS - ageMs);
        sessions.push({
            student_id:       studentId,
            message_count:    data.messages.length,
            age_hours:        +(ageMs  / 3_600_000).toFixed(2),
            idle_minutes:     +(idleMs / 60_000).toFixed(2),
            expires_in_hours: +(expiresMs / 3_600_000).toFixed(2),
        });
    }
    res.json({ active_sessions: sessions.length, sessions });
});

// ──────────────────────────────────────────────
// WhatsApp Session Helpers
// ──────────────────────────────────────────────
function openSession(jid, sock) {
    const old = activeSessions.get(jid);
    if (old) clearTimeout(old.timer);

    const timer = setTimeout(async () => {
        activeSessions.delete(jid);
        console.log(`[Session] Auto-closed (inactivity): ${jid}`);
        try { await sock.sendMessage(jid, { text: INACTIVITY_GOODBYE }); } catch (_) {}
    }, INACTIVITY_TIMEOUT);

    activeSessions.set(jid, { timer, activatedAt: new Date() });
    console.log(`[Session] Opened: ${jid}`);
}

function resetTimer(jid, sock) {
    const session = activeSessions.get(jid);
    if (!session) return;
    clearTimeout(session.timer);
    session.timer = setTimeout(async () => {
        activeSessions.delete(jid);
        console.log(`[Session] Auto-closed (inactivity): ${jid}`);
        try { await sock.sendMessage(jid, { text: INACTIVITY_GOODBYE }); } catch (_) {}
    }, INACTIVITY_TIMEOUT);
}

async function closeSession(jid, sock) {
    const session = activeSessions.get(jid);
    if (!session) return;
    clearTimeout(session.timer);
    activeSessions.delete(jid);
    console.log(`[Session] Closed: ${jid}`);
    if (sock) {
        try { await sock.sendMessage(jid, { text: GOODBYE_MESSAGE }); } catch (_) {}
    }
}

function isSessionActive(jid) {
    return activeSessions.has(jid);
}

// ──────────────────────────────────────────────
// Message Helpers
// ──────────────────────────────────────────────
function extractText(msg) {
    const m = msg?.message;
    if (!m) return '';
    const inner = normalizeMessageContent(m) || m;
    return (
        inner.conversation?.trim()              ||
        inner.extendedTextMessage?.text?.trim() ||
        inner.imageMessage?.caption?.trim()     ||
        inner.videoMessage?.caption?.trim()     ||
        ''
    );
}

function shouldSkipBotMessage(msg) {
    if (!msg?.key) return true;
    if (msg.key.fromMe) return true;

    const jids = [msg.key.remoteJid, msg.key.remoteJidAlt].filter(Boolean).map(String);
    if (jids.length === 0) return true;

    for (const j of jids) {
        if (j === 'status@broadcast')  return true;
        if (j.endsWith('@broadcast'))   return true;
        if (j.endsWith('@g.us'))        return true;
        if (j.includes('@newsletter'))  return true;
    }

    if (msg.messageStubType != null) return true;

    const raw = msg.message;
    if (!raw) return true;

    const inner   = normalizeMessageContent(raw) || raw;
    if (inner.call) return true;

    const topKeys = Object.keys(inner);
    if (topKeys.some((k) => k === 'call' || k.includes('scheduledCall') || k.endsWith('CallMessage'))) {
        return true;
    }

    return false;
}

// ──────────────────────────────────────────────
// WhatsApp Bot
// ──────────────────────────────────────────────
let waSocket = null;

async function startMufaroBot() {
    await fs.mkdir(BOT_SESSION_DIR, { recursive: true });

    if (waSocket) {
        try {
            waSocket.ev.removeAllListeners();
            await waSocket.end();
        } catch (_) {}
        waSocket = null;
    }

    const { state, saveCreds } = await useMultiFileAuthState(BOT_SESSION_DIR);
    const { version }          = await fetchLatestBaileysVersion();
    const logger               = pino({ level: 'silent' });

    const sock = makeWASocket({
        version,
        logger,
        browser: ['Mufaro Bot', 'Chrome', '1.0.0'],
        auth: {
            creds: state.creds,
            keys:  makeCacheableSignalKeyStore(
                state.keys,
                pino({ level: 'fatal' }).child({ level: 'fatal' })
            ),
        },
        generateHighQualityLinkPreview: false,
        defaultQueryTimeoutMs:          60_000,
        connectTimeoutMs:               60_000,
        keepAliveIntervalMs:            30_000,
        markOnlineOnConnect:            true,
    });

    waSocket = sock;

    sock.ev.on('creds.update', saveCreds);

    sock.ev.on('connection.update', async (update) => {
        const { connection, lastDisconnect, qr } = update;

        if (qr) {
            try {
                botQrDataUrl = await QRCode.toDataURL(qr, { margin: 2, width: 400 });
                console.log('\n');
                console.log('╔══════════════════════════════════════════════╗');
                console.log('║   📱 SCAN THIS QR CODE WITH WHATSAPP        ║');
                console.log('║   WhatsApp → Linked Devices → Link a Device ║');
                console.log('╚══════════════════════════════════════════════╝');
                console.log('');
                const qrTerminal = await QRCode.toString(qr, { type: 'terminal', small: true });
                console.log(qrTerminal);
                const host = process.env.PUBLIC_HOST || `http://localhost:${PORT}`;
                console.log(`   Or open in browser: ${host.replace(/\/$/, '')}/bot-qr\n`);
            } catch (e) {
                console.error('Failed to render QR code:', e);
            }
        }

        if (connection === 'open') {
            botQrDataUrl = null;
            console.log('');
            console.log('✅ Mufaro WhatsApp bot connected successfully!');
            console.log('   Bot is now listening for messages.');
            console.log(`   Stop words: ${STOP_WORDS.join(', ')}`);
            console.log(`   Inactivity timeout: ${INACTIVITY_TIMEOUT / 60_000} min`);
            console.log('');
        }

        if (connection === 'close') {
            const code            = lastDisconnect?.error?.output?.statusCode;
            const shouldReconnect = code !== DisconnectReason.loggedOut;
            console.log(`WhatsApp connection closed (${code}). Reconnect: ${shouldReconnect}`);
            if (shouldReconnect) {
                const waitMs = code === DisconnectReason.restartRequired ? 5000 : 3000;
                await delay(waitMs);
                startMufaroBot().catch((e) => console.error('Reconnect failed:', e));
            }
        }
    });

    // ── Main Message Handler ──────────────────
    sock.ev.on('messages.upsert', async ({ messages, type }) => {
        if (type !== 'notify') return;

        const msg = messages[0];
        if (shouldSkipBotMessage(msg)) return;

        const jid = msg.key.remoteJid;
        if (!jid) return;

        const text = extractText(msg);
        if (!text) return;

        const textLower = text.trim().toLowerCase();
        const sessionOn = isSessionActive(jid);

        console.log(`📩 [${sessionOn ? 'ACTIVE' : 'NEW'}] ${jid}: ${text.slice(0, 100)}`);

        // ── CASE 1: Stop word → end session ──
        if (STOP_WORDS.includes(textLower)) {
            await closeSession(jid, sock);
            return;
        }

        // ── CASE 2: New contact — greet, then answer ─────────
        if (!sessionOn) {
            openSession(jid, sock);
            try {
                await sock.sendPresenceUpdate('composing', jid);
                const greeting = await generateGreeting(msg.pushName);
                await sock.sendPresenceUpdate('paused', jid);
                await sock.sendMessage(jid, { text: greeting });
                await delay(500);
            } catch (err) {
                console.error('Failed to send greeting:', err);
            }
        } else {
            resetTimer(jid, sock);
        }

        // ── CASE 3: Answer every message via Groq ────────────
        try {
            await sock.sendPresenceUpdate('composing', jid);
            const reply = await askMufaro(jid, text);
            await sock.sendPresenceUpdate('paused', jid);
            await sock.sendMessage(jid, { text: reply });
        } catch (err) {
            console.error('Mufaro / reply error:', err);
            try {
                await sock.sendMessage(jid, {
                    text: '⚠️ Sorry, I could not reach the assistant right now. Please try again in a moment.',
                });
            } catch (_) {}
        }
    });
}

// ──────────────────────────────────────────────
// Start Server
// ──────────────────────────────────────────────
const server = app.listen(PORT, '0.0.0.0', () => {
    console.log('\n===== Mufaro Bot Starting =====');
    console.log(`Server:  http://0.0.0.0:${PORT}`);
    console.log(`Model:   llama-3.3-70b-versatile (via Groq)`);
    console.log(`Mode:    Dedicated number — responds to every message`);
    console.log(`Timeout: ${INACTIVITY_TIMEOUT / 60_000} min inactivity closes session`);
    console.log(`Session: ${BOT_SESSION_DIR}`);
    console.log('');
    console.log('Connecting to WhatsApp... wait for the QR code below.\n');

    startMufaroBot().catch((e) => console.error('WhatsApp bot failed to start:', e));
});

server.on('error', (err) => {
    console.error('Server error:', err);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    process.exit(1);
});

export default app;