'use client';
/**
 * ArthSathi — Adaptive Onboarding Page v3
 * 
 * - 39 questions driven entirely from i18n/questions.json
 * - Full branching: depends_on conditions skip irrelevant questions
 * - All profile fields collected and sent to backend
 * - Voice mode: TTS reads each question, mic records answer
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { LANGUAGES, getSavedLanguage, getSavedMode, setLanguage as persistLang, setMode as persistMode, type LangCode } from '../../lib/i18n';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

// Indian states
const STATES = [
  'Andhra Pradesh','Arunachal Pradesh','Assam','Bihar','Chhattisgarh',
  'Goa','Gujarat','Haryana','Himachal Pradesh','Jharkhand','Karnataka',
  'Kerala','Madhya Pradesh','Maharashtra','Manipur','Meghalaya','Mizoram',
  'Nagaland','Odisha','Punjab','Rajasthan','Sikkim','Tamil Nadu','Telangana',
  'Tripura','Uttar Pradesh','Uttarakhand','West Bengal',
  'Delhi','Jammu and Kashmir','Ladakh','Puducherry','Chandigarh',
  'Andaman and Nicobar Islands','Dadra and Nagar Haveli','Lakshadweep',
];

// Full profile type with all fields
interface Profile {
  preferred_language: LangCode;
  interaction_mode: 'text' | 'voice';
  state: string; district: string;
  age: string; gender: string; marital_status: string; is_widow: boolean | null;
  caste_category: string; religion: string;
  occupation: string; sub_occupation: string; land_holding_acres: string;
  has_kisan_credit_card: boolean | null;
  monthly_income: string; family_monthly_income: string;
  num_dependents: string; total_debts: string;
  ration_card_type: string;
  has_bank_account: boolean | null; has_jan_dhan_account: boolean | null;
  has_aadhaar: boolean | null; has_kisan_credit_card_2: boolean | null;
  has_disability: boolean | null; disability_type: string;
  disability_percentage: string; has_disability_certificate: boolean | null;
  has_girl_child: boolean | null; num_girl_children: string;
  is_student: boolean | null; education_level: string; student_class_level: string;
  is_senior_citizen: boolean | null; is_minority: boolean | null;
  residence_type: string;
  financial_goal: string; goal_description: string;
  goal_amount: string; goal_timeframe_months: string; urgency_level: string;
}

const EMPTY: Profile = {
  preferred_language: 'hi', interaction_mode: 'voice',
  state: '', district: '', age: '', gender: '', marital_status: '',
  is_widow: null, caste_category: '', religion: '', occupation: '',
  sub_occupation: '', land_holding_acres: '', has_kisan_credit_card: null,
  monthly_income: '', family_monthly_income: '', num_dependents: '',
  total_debts: '', ration_card_type: '',
  has_bank_account: null, has_jan_dhan_account: null, has_aadhaar: null,
  has_kisan_credit_card_2: null, has_disability: null, disability_type: '',
  disability_percentage: '', has_disability_certificate: null,
  has_girl_child: null, num_girl_children: '', is_student: null,
  education_level: '', student_class_level: '', is_senior_citizen: null,
  is_minority: null, residence_type: '',
  financial_goal: '', goal_description: '', goal_amount: '',
  goal_timeframe_months: '', urgency_level: '',
};

type StepType = 'lang_select'|'mode_select'|'state_select'|'text'|'number'|
                'choice'|'choice_grid'|'yesno'|'income_band'|'ration_card'|
                'goal_select'|'urgency_scale';

interface StepDef {
  id: keyof Profile;
  type: StepType;
  icon: string;
  required?: boolean;
  show?: (p: Profile) => boolean;
  options?: { value: string|number; icon?: string; color?: string }[];
}

// Step definitions matching questions.json
const STEPS: StepDef[] = [
  { id: 'preferred_language', type: 'lang_select',  icon: '🌐', required: true },
  { id: 'interaction_mode',   type: 'mode_select',  icon: '🎤', required: true },
  { id: 'state',              type: 'state_select', icon: '📍', required: true },
  { id: 'district',           type: 'text',         icon: '🏘️' },
  { id: 'age',                type: 'number',       icon: '🎂', required: true },
  { id: 'gender',             type: 'choice',       icon: '👤', required: true,
    options: [{value:'male',icon:'👨'},{value:'female',icon:'👩'},{value:'transgender',icon:'🏳️‍⚧️'},{value:'prefer_not_to_say',icon:'🔒'}] },
  { id: 'marital_status',     type: 'choice',       icon: '💍',
    options: [{value:'single',icon:'🙋'},{value:'married',icon:'💑'},{value:'widowed',icon:'💙'},{value:'divorced',icon:'📋'},{value:'separated',icon:'🚪'}] },
  { id: 'is_widow',           type: 'yesno',        icon: '💙',
    show: p => p.marital_status === 'widowed' },
  { id: 'caste_category',     type: 'choice',       icon: '📋', required: true,
    options: [{value:'general',icon:'⭕'},{value:'obc',icon:'🟡'},{value:'sc',icon:'🔵'},{value:'st',icon:'🟢'},{value:'ews',icon:'🟠'},{value:'nt',icon:'🟣'},{value:'other',icon:'⚪'}] },
  { id: 'religion',           type: 'choice',       icon: '🕊️',
    options: [{value:'hindu',icon:'🔱'},{value:'muslim',icon:'☪️'},{value:'christian',icon:'✝️'},{value:'sikh',icon:'☬'},{value:'buddhist',icon:'☸️'},{value:'jain',icon:'🟤'},{value:'parsi',icon:'🔥'},{value:'other',icon:'🌏'},{value:'prefer_not_to_say',icon:'🔒'}] },
  { id: 'occupation',         type: 'choice_grid',  icon: '💼', required: true,
    options: [{value:'farmer',icon:'🌾'},{value:'marginal_farmer',icon:'🌱'},{value:'farm_laborer',icon:'🚜'},{value:'fisherman',icon:'🎣'},{value:'weaver',icon:'🧵'},{value:'artisan',icon:'🔨'},{value:'self_employed',icon:'🛒'},{value:'street_vendor',icon:'🛍️'},{value:'daily_wage',icon:'👷'},{value:'construction_worker',icon:'🧱'},{value:'domestic_worker',icon:'🏠'},{value:'small_trader',icon:'🏪'},{value:'student',icon:'📚'},{value:'homemaker',icon:'👪'},{value:'unemployed',icon:'🔍'},{value:'other',icon:'💼'}] },
  { id: 'sub_occupation',     type: 'text',         icon: '🔧',
    show: p => ['artisan','weaver','self_employed','small_trader','fisherman'].includes(p.occupation) },
  { id: 'land_holding_acres', type: 'number',       icon: '🌾',
    show: p => ['farmer','marginal_farmer'].includes(p.occupation) },
  { id: 'has_kisan_credit_card', type: 'yesno',     icon: '💳',
    show: p => ['farmer','marginal_farmer'].includes(p.occupation) },
  { id: 'monthly_income',     type: 'income_band',  icon: '💰', required: true,
    options: [{value:'0',icon:'0️⃣'},{value:'1000',icon:'1️⃣'},{value:'2500',icon:'2️⃣'},{value:'5000',icon:'3️⃣'},{value:'8000',icon:'4️⃣'},{value:'12000',icon:'5️⃣'},{value:'20000',icon:'6️⃣'},{value:'35000',icon:'7️⃣'}] },
  { id: 'family_monthly_income', type: 'income_band', icon: '👨‍👩‍👧‍👦',
    options: [{value:'0',icon:'0️⃣'},{value:'3000',icon:'1️⃣'},{value:'8000',icon:'2️⃣'},{value:'15000',icon:'3️⃣'},{value:'25000',icon:'4️⃣'},{value:'50000',icon:'5️⃣'},{value:'100000',icon:'6️⃣'}] },
  { id: 'num_dependents',     type: 'number',       icon: '👨‍👩‍👧' },
  { id: 'ration_card_type',   type: 'ration_card',  icon: '🪪', required: true,
    options: [{value:'aay',color:'#FFD700',icon:'🟡'},{value:'phh',color:'#FF6B6B',icon:'🔴'},{value:'nphh',color:'#90EE90',icon:'🟢'},{value:'annapurna',color:'#87CEEB',icon:'🔵'},{value:'state_other',color:'#DDA0DD',icon:'🟣'},{value:'none',color:'#D3D3D3',icon:'⚪'},{value:'applied',color:'#FFA500',icon:'🟠'}] },
  { id: 'has_bank_account',   type: 'yesno', icon: '🏦', required: true },
  { id: 'has_jan_dhan_account', type: 'yesno', icon: '🏦', show: p => p.has_bank_account === true },
  { id: 'has_aadhaar',        type: 'yesno', icon: '🪪', required: true },
  { id: 'has_disability',     type: 'yesno', icon: '♿', required: true },
  { id: 'disability_type',    type: 'choice', icon: '🩺',
    show: p => p.has_disability === true,
    options: [{value:'visual',icon:'👁️'},{value:'hearing',icon:'👂'},{value:'locomotor',icon:'🦽'},{value:'mental_illness',icon:'🧠'},{value:'intellectual',icon:'📖'},{value:'multiple',icon:'➕'},{value:'other',icon:'❓'}] },
  { id: 'disability_percentage', type: 'number', icon: '📊', show: p => p.has_disability === true },
  { id: 'has_disability_certificate', type: 'yesno', icon: '📄', show: p => p.has_disability === true },
  { id: 'has_girl_child',     type: 'yesno', icon: '👧', show: p => p.gender === 'female' },
  { id: 'num_girl_children',  type: 'number', icon: '👧', show: p => p.has_girl_child === true },
  { id: 'is_student',         type: 'yesno', icon: '📚', show: p => parseInt(p.age||'99') <= 35 },
  { id: 'education_level',    type: 'choice', icon: '🎓',
    show: p => p.is_student === true || p.occupation === 'student',
    options: [{value:'illiterate',icon:'📝'},{value:'primary',icon:'🔢'},{value:'middle',icon:'📖'},{value:'matriculation',icon:'📗'},{value:'higher_secondary',icon:'📘'},{value:'graduate',icon:'🎓'},{value:'postgraduate',icon:'🏅'},{value:'professional',icon:'⚕️'}] },
  { id: 'student_class_level', type: 'text', icon: '🏫', show: p => p.is_student === true },
  { id: 'is_senior_citizen',  type: 'yesno', icon: '👴', show: p => parseInt(p.age||'0') >= 55 },
  { id: 'is_minority',        type: 'yesno', icon: '🌍',
    show: p => ['muslim','christian','sikh','buddhist','jain','parsi'].includes(p.religion) },
  { id: 'residence_type',     type: 'choice', icon: '🏘️', required: true,
    options: [{value:'rural',icon:'🌾'},{value:'semi_urban',icon:'🏘️'},{value:'urban',icon:'🏙️'}] },
  { id: 'financial_goal',     type: 'goal_select', icon: '🎯', required: true,
    options: [{value:'farming',icon:'🌾'},{value:'housing',icon:'🏠'},{value:'health',icon:'🏥'},{value:'education',icon:'📚'},{value:'employment',icon:'💼'},{value:'insurance',icon:'🛡️'},{value:'pension',icon:'👴'},{value:'business',icon:'🏪'},{value:'daughter',icon:'👧'},{value:'loan',icon:'💳'},{value:'savings',icon:'💰'},{value:'disability_support',icon:'♿'},{value:'water_sanitation',icon:'💧'},{value:'skill_training',icon:'🔧'},{value:'other',icon:'⭐'}] },
  { id: 'goal_description',   type: 'text', icon: '✍️' },
  { id: 'goal_amount',        type: 'number', icon: '💵' },
  { id: 'goal_timeframe_months', type: 'choice', icon: '📅',
    options: [{value:'1',icon:'⚡'},{value:'3',icon:'📅'},{value:'6',icon:'🗓️'},{value:'12',icon:'📆'},{value:'24',icon:'🗓️'},{value:'60',icon:'🏁'}] },
  { id: 'total_debts', type: 'number', icon: '💸',
    show: p => ['loan','business','farming','housing'].includes(p.financial_goal) },
  { id: 'urgency_level',      type: 'urgency_scale', icon: '⚡', required: true,
    options: [{value:1,icon:'😌'},{value:2,icon:'🙁'},{value:3,icon:'😟'},{value:4,icon:'😰'},{value:5,icon:'🚨'}] },
];

function getActiveSteps(p: Profile) {
  return STEPS.filter(s => !s.show || s.show(p));
}

type QData = Record<string, unknown>;
function qText(data: QData, id: string, field: string, fallback = ''): string {
  const s = data[id];
  if (!s || typeof s !== 'object') return fallback;
  const v = (s as Record<string, unknown>)[field];
  return typeof v === 'string' ? v : fallback;
}
function qOptions(data: QData, id: string): Record<string, string> {
  const s = data[id];
  if (!s || typeof s !== 'object') return {};
  const opts = (s as Record<string, unknown>).options;
  if (!opts || typeof opts !== 'object') return {};
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(opts as Record<string, unknown>)) {
    if (typeof v === 'string') out[k] = v;
    else if (v && typeof v === 'object' && 'label' in (v as object))
      out[k] = String((v as Record<string, unknown>).label);
  }
  return out;
}
function common(data: QData, key: string): string {
  const c = data.common;
  if (!c || typeof c !== 'object') return key;
  const v = (c as Record<string, unknown>)[key];
  return typeof v === 'string' ? v : key;
}

export default function OnboardingPage() {
  const router = useRouter();
  const [profile, setProfile] = useState<Profile>({ ...EMPTY });
  const [stepIdx, setStepIdx] = useState(0);
  const [qData,   setQData]   = useState<QData>({});
  const [submitting, setSub]  = useState(false);
  const [error,   setError]   = useState('');
  const [recording, setRec]   = useState(false);
  const [voiceTxt, setVoiceTxt] = useState('');
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  // Load translations
  useEffect(() => {
    const lang = profile.preferred_language;
    fetch(`${API}/api/i18n/${lang}/questions`)
      .then(r => r.ok ? r.json() : fetch(`${API}/api/i18n/en/questions`).then(r2 => r2.json()))
      .then(setQData)
      .catch(() => {});
  }, [profile.preferred_language]);

  // Restore progress
  useEffect(() => {
    const saved = localStorage.getItem('as_profile');
    if (saved) { try { setProfile({ ...EMPTY, ...JSON.parse(saved) }); } catch {} }
    setProfile(p => ({
      ...p,
      preferred_language: getSavedLanguage(),
      interaction_mode: getSavedMode(),
    }));
  }, []);

  useEffect(() => { localStorage.setItem('as_profile', JSON.stringify(profile)); }, [profile]);

  const activeSteps = getActiveSteps(profile);
  const step = activeSteps[stepIdx];
  const isLast = stepIdx === activeSteps.length - 1;
  const progress = ((stepIdx + 1) / activeSteps.length) * 100;

  // TTS: read question aloud in voice mode
  useEffect(() => {
    if (profile.interaction_mode === 'voice' && qData && step) {
      const text = qText(qData, step.id, 'question');
      if (text && typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        const locales: Record<LangCode, string> = { hi:'hi-IN', mr:'mr-IN', ta:'ta-IN', te:'te-IN', bn:'bn-IN', en:'en-IN' };
        u.lang = locales[profile.preferred_language] || 'hi-IN';
        u.rate = 0.85;
        window.speechSynthesis.speak(u);
      }
    }
  }, [stepIdx, qData]); // eslint-disable-line

  const update = useCallback((field: keyof Profile, val: unknown) => {
    setProfile(p => ({ ...p, [field]: val }));
    setError('');
  }, []);

  const handleNext = async () => {
    if (step.required) {
      const val = profile[step.id];
      if (val === '' || val === null || val === undefined) {
        setError('Please answer this question to continue.');
        return;
      }
    }
    if (isLast) await submit();
    else setStepIdx(i => i + 1);
  };

  const submit = async () => {
    setSub(true);
    try {
      const monthly = parseFloat(profile.monthly_income) || 0;
      const familyMonthly = parseFloat(profile.family_monthly_income) || 0;

      const uRes = await fetch(`${API}/api/users`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          full_name: undefined,
          state: profile.state, district: profile.district,
          preferred_language: profile.preferred_language,
          interaction_mode: profile.interaction_mode,
          residence_type: profile.residence_type || null,
        }),
      });
      if (!uRes.ok) throw new Error('Could not save user');
      const uData = await uRes.json();
      const userId = uData.data.id;
      localStorage.setItem('as_user_id', userId);

      const pRes = await fetch(`${API}/api/users/${userId}/profile`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          age: parseInt(profile.age) || null,
          gender: profile.gender || null,
          marital_status: profile.marital_status || null,
          religion: profile.religion || null,
          caste_category: profile.caste_category || null,
          occupation: profile.occupation || null,
          sub_occupation: profile.sub_occupation || null,
          monthly_income: monthly,
          annual_income: monthly * 12,
          family_monthly_income: familyMonthly || null,
          family_annual_income: familyMonthly ? familyMonthly * 12 : null,
          total_debts: parseFloat(profile.total_debts) || 0,
          land_holding_acres: parseFloat(profile.land_holding_acres) || 0,
          num_dependents: parseInt(profile.num_dependents) || 0,
          is_bpl: ['aay','phh'].includes(profile.ration_card_type),
          ration_card_type: profile.ration_card_type || null,
          has_bank_account: profile.has_bank_account,
          has_jan_dhan_account: profile.has_jan_dhan_account,
          has_aadhaar: profile.has_aadhaar,
          has_kisan_credit_card: profile.has_kisan_credit_card,
          has_disability: profile.has_disability,
          disability_type: profile.disability_type || null,
          disability_percentage: parseInt(profile.disability_percentage) || null,
          has_disability_certificate: profile.has_disability_certificate,
          has_girl_child: profile.has_girl_child,
          num_girl_children: parseInt(profile.num_girl_children) || 0,
          is_widow: profile.is_widow || profile.marital_status === 'widowed',
          is_student: profile.is_student || profile.occupation === 'student',
          education_level: profile.education_level || null,
          student_class_level: profile.student_class_level || null,
          is_senior_citizen: profile.is_senior_citizen || parseInt(profile.age||'0') >= 60,
          is_minority: profile.is_minority,
          residence_type: profile.residence_type || null,
          financial_goal: profile.financial_goal,
          goal_description: profile.goal_description || null,
          goal_amount: parseFloat(profile.goal_amount) || null,
          goal_timeframe_months: parseInt(profile.goal_timeframe_months) || null,
          urgency_level: parseInt(profile.urgency_level) || 3,
        }),
      });
      if (!pRes.ok) throw new Error('Could not save profile');
      router.push('/results');
    } catch {
      localStorage.setItem('as_offline_profile', JSON.stringify(profile));
      router.push('/results?offline=1');
    } finally { setSub(false); }
  };

  // Voice recording
  const startRec = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
      const rec = new MediaRecorder(stream, { mimeType: mime });
      chunksRef.current = [];
      rec.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      rec.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        const fd = new FormData();
        fd.append('audio', new Blob(chunksRef.current, { type: mime }), 'rec.webm');
        fd.append('language', profile.preferred_language);
        try {
          const r = await fetch(`${API}/api/voice/stt`, { method: 'POST', body: fd });
          const d = await r.json();
          if (d.text?.trim()) setVoiceTxt(d.text.trim());
        } catch { setError('Voice not understood. Please type.'); }
        setRec(false);
      };
      rec.start(); mediaRef.current = rec; setRec(true);
    } catch { setError('Microphone access denied.'); }
  };
  const stopRec = () => mediaRef.current?.stop();

  if (!step) return null;

  const questionText = qText(qData, step.id, 'question', step.id);
  const hintText     = qText(qData, step.id, 'hint', '');
  const placeholder  = qText(qData, step.id, 'placeholder', '');
  const options      = qOptions(qData, step.id);
  const stepLabel    = (common(qData,'step')||'Step {{current}} of {{total}}')
    .replace('{{current}}', String(stepIdx+1)).replace('{{total}}', String(activeSteps.length));

  return (
    <main className="min-h-screen bg-white flex flex-col max-w-lg mx-auto">
      {/* Progress */}
      <div className="h-2 bg-gray-100 sticky top-0 z-10">
        <div className="h-2 bg-primary transition-all duration-500" style={{ width: `${progress}%` }} />
      </div>

      {/* Header */}
      <header className="flex items-center justify-between px-4 pt-4 pb-2">
        <span className="text-gray-400 text-sm font-medium">{stepLabel}</span>
        {profile.interaction_mode === 'voice' && (
          <button onClick={() => {
            if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
              window.speechSynthesis.cancel();
              const u = new SpeechSynthesisUtterance(questionText);
              const locs: Record<LangCode, string> = { hi:'hi-IN', mr:'mr-IN', ta:'ta-IN', te:'te-IN', bn:'bn-IN', en:'en-IN' };
              u.lang = locs[profile.preferred_language] || 'hi-IN';
              window.speechSynthesis.speak(u);
            }
          }} className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center text-primary text-lg" aria-label="Replay">
            🔊
          </button>
        )}
      </header>

      {/* Question */}
      <div className="flex-1 px-5 pt-4 pb-4 flex flex-col">
        <div className="flex items-start gap-3 mb-5">
          <span className="text-4xl flex-shrink-0">{step.icon}</span>
          <div>
            <h2 className="text-2xl font-bold text-gray-900 leading-snug">{questionText}</h2>
            {hintText && <p className="text-gray-500 text-sm mt-1">{hintText}</p>}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          <StepInput
            step={step} profile={profile} options={options}
            qData={qData} onChange={update}
            yes={common(qData,'yes')||'Yes'} no={common(qData,'no')||'No'}
            placeholder={placeholder}
          />
        </div>

        {/* Voice mic for text/number inputs */}
        {profile.interaction_mode === 'voice' && ['text','number'].includes(step.type) && (
          <div className="mt-4 flex flex-col items-center gap-2">
            <button
              onPointerDown={startRec} onPointerUp={stopRec}
              className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl text-white shadow transition-transform active:scale-95 ${recording ? 'bg-red-500 voice-recording' : 'bg-primary'}`}
              aria-label={recording ? 'Stop' : 'Speak'}
            >{recording ? '🔴' : '🎤'}</button>
            {voiceTxt && (
              <div className="w-full bg-gray-50 border rounded-2xl p-3 flex items-center justify-between gap-3">
                <span className="text-sm text-gray-800">"{voiceTxt}"</span>
                <button className="bg-primary text-white px-3 py-1.5 rounded-xl text-sm font-semibold"
                  onClick={() => { update(step.id as keyof Profile, voiceTxt); setVoiceTxt(''); }}>
                  Use ✓
                </button>
              </div>
            )}
          </div>
        )}

        {error && <p className="mt-3 text-red-600 text-sm font-medium" role="alert">{error}</p>}
      </div>

      {/* Navigation */}
      <div className="px-5 pb-6 pt-2 space-y-2 border-t border-gray-100">
        <button onClick={handleNext} disabled={submitting} className="btn-primary">
          {submitting ? 'Saving...' : isLast ? (common(qData,'finish')||'Show My Schemes') : (common(qData,'next')||'Next')}
        </button>
        {stepIdx > 0 && (
          <button onClick={() => setStepIdx(i => Math.max(0,i-1))} disabled={submitting} className="btn-secondary">
            {common(qData,'back')||'Back'}
          </button>
        )}
        {!step.required && (
          <button onClick={() => isLast ? submit() : setStepIdx(i => i+1)} className="w-full py-2 text-gray-400 text-sm text-center">
            {common(qData,'skip')||'Skip'}
          </button>
        )}
      </div>
    </main>
  );
}

// ─── StepInput ────────────────────────────────────────────────────────────────

interface SIProps {
  step: StepDef; profile: Profile; options: Record<string,string>;
  qData: QData; onChange: (f: keyof Profile, v: unknown) => void;
  yes: string; no: string; placeholder: string;
}

function StepInput({ step, profile, options, qData, onChange, yes, no, placeholder }: SIProps) {
  const id = step.id; const val = profile[id];

  if (step.type === 'lang_select') return (
    <div className="grid grid-cols-2 gap-3">
      {(Object.entries(LANGUAGES) as [LangCode, {native:string}][]).map(([code, info]) => (
        <button key={code}
          onClick={() => { onChange(id, code); persistLang(code); }}
          className={`py-5 rounded-2xl border-2 text-xl font-bold transition-all ${val===code?'border-primary bg-primary/10 text-primary':'border-gray-200 text-gray-700'}`}>
          {info.native}
        </button>
      ))}
    </div>
  );

  if (step.type === 'mode_select') {
    const modeOpts = [
      { value:'text',  icon:'⌨️', label: qOptions(qData,'interaction_mode').text  || 'Type & Read',    desc:'Type your answers' },
      { value:'voice', icon:'🎤', label: qOptions(qData,'interaction_mode').voice || 'Speak & Listen', desc:'Speak your answers' },
    ];
    return (
      <div className="space-y-4">
        {modeOpts.map(o => (
          <button key={o.value}
            onClick={() => { onChange(id, o.value); persistMode(o.value as 'text'|'voice'); }}
            className={`w-full flex items-center gap-4 p-5 rounded-2xl border-2 text-left transition-all ${val===o.value?'border-primary bg-primary/10':'border-gray-200'}`}>
            <span className="text-5xl">{o.icon}</span>
            <div>
              <p className="font-bold text-xl text-gray-900">{o.label}</p>
              <p className="text-gray-500 text-base">{o.desc}</p>
            </div>
          </button>
        ))}
      </div>
    );
  }

  if (step.type === 'state_select') return (
    <select className="form-input" value={String(val||'')} onChange={e => onChange(id, e.target.value)}>
      <option value="">— Select State —</option>
      {STATES.map(s => <option key={s} value={s}>{s}</option>)}
    </select>
  );

  if (step.type === 'yesno') return (
    <div className="flex gap-4">
      {[{v:true,label:yes,icon:'✅'},{v:false,label:no,icon:'❌'}].map(o => (
        <button key={String(o.v)}
          onClick={() => onChange(id, o.v)}
          className={`flex-1 py-6 rounded-2xl border-2 flex flex-col items-center gap-2 transition-all ${val===o.v?'border-primary bg-primary/10':'border-gray-200'}`}>
          <span className="text-4xl">{o.icon}</span>
          <span className="font-bold text-xl text-gray-900">{o.label}</span>
        </button>
      ))}
    </div>
  );

  if (step.type === 'income_band') {
    const incOpts = options;
    const ICONS: Record<string,string> = {'0':'0️⃣','1000':'1️⃣','2500':'2️⃣','5000':'3️⃣','8000':'4️⃣','12000':'5️⃣','20000':'6️⃣','35000':'7️⃣','3000':'1️⃣','15000':'3️⃣','25000':'4️⃣','50000':'5️⃣','100000':'6️⃣'};
    return (
      <div className="space-y-2">
        {Object.entries(incOpts).map(([v, label]) => (
          <button key={v} onClick={() => onChange(id, v)}
            className={`w-full flex items-center gap-4 p-4 rounded-2xl border-2 text-left transition-all ${val===v?'border-primary bg-primary/10':'border-gray-200'}`}>
            <span className="text-3xl">{ICONS[v]||'💰'}</span>
            <span className="text-lg font-medium text-gray-900">{label}</span>
          </button>
        ))}
      </div>
    );
  }

  if (step.type === 'ration_card') {
    const CARD_COLORS: Record<string,string> = {
      aay:'bg-yellow-100 border-yellow-400', phh:'bg-red-100 border-red-400',
      nphh:'bg-green-100 border-green-400', annapurna:'bg-blue-100 border-blue-400',
      state_other:'bg-purple-100 border-purple-400', none:'bg-gray-100 border-gray-400',
      applied:'bg-orange-100 border-orange-400',
    };
    return (
      <div className="space-y-2">
        {Object.entries(options).map(([v, label]) => (
          <button key={v} onClick={() => onChange(id, v)}
            className={`w-full flex items-center gap-3 p-4 rounded-2xl border-2 text-left transition-all ${CARD_COLORS[v]||'border-gray-200'} ${val===v?'ring-2 ring-primary':''}`}>
            <span className="text-2xl">{step.options?.find(o=>o.value===v)?.icon||'🪪'}</span>
            <span className="text-base font-medium text-gray-900">{label}</span>
          </button>
        ))}
      </div>
    );
  }

  if (step.type === 'choice_grid') {
    return (
      <div className="grid grid-cols-2 gap-2">
        {step.options?.map(o => (
          <button key={String(o.value)} onClick={() => onChange(id, o.value)}
            className={`flex items-center gap-2 p-4 rounded-2xl border-2 text-left transition-all ${val===o.value?'border-primary bg-primary/10':'border-gray-200'}`}>
            <span className="text-2xl flex-shrink-0">{o.icon}</span>
            <span className="text-sm font-medium text-gray-900 leading-tight">{options[String(o.value)]||String(o.value)}</span>
          </button>
        ))}
      </div>
    );
  }

  if (step.type === 'goal_select') return (
    <div className="grid grid-cols-2 gap-2">
      {step.options?.map(o => (
        <button key={String(o.value)} onClick={() => onChange(id, o.value)}
          className={`flex flex-col items-center justify-center p-4 rounded-2xl border-2 text-center transition-all ${val===o.value?'border-primary bg-primary/10':'border-gray-200'}`}>
          <span className="text-3xl mb-1">{o.icon}</span>
          <span className="text-xs font-medium text-gray-900 leading-tight">{options[String(o.value)]||String(o.value)}</span>
        </button>
      ))}
    </div>
  );

  if (step.type === 'urgency_scale') {
    const UCOLS = ['bg-green-100 border-green-400','bg-lime-100 border-lime-400','bg-yellow-100 border-yellow-400','bg-orange-100 border-orange-400','bg-red-100 border-red-400'];
    return (
      <div className="space-y-2">
        {step.options?.map((o, i) => (
          <button key={String(o.value)} onClick={() => onChange(id, String(o.value))}
            className={`w-full flex items-center gap-4 p-4 rounded-2xl border-2 text-left transition-all ${UCOLS[i]} ${val===String(o.value)?'ring-2 ring-primary':''}`}>
            <span className="text-3xl">{o.icon}</span>
            <span className="text-base font-medium text-gray-900">{options[String(o.value)]||String(o.value)}</span>
          </button>
        ))}
      </div>
    );
  }

  if (step.type === 'choice') {
    return (
      <div className="space-y-2">
        {step.options?.map(o => (
          <button key={String(o.value)} onClick={() => onChange(id, o.value)}
            className={`w-full flex items-center gap-4 p-4 rounded-2xl border-2 text-left transition-all ${val===o.value?'border-primary bg-primary/10':'border-gray-200'}`}>
            <span className="text-3xl">{o.icon}</span>
            <span className="text-lg font-medium text-gray-900">{options[String(o.value)]||String(o.value)}</span>
          </button>
        ))}
      </div>
    );
  }

  // text / number
  return (
    <input type={step.type==='number'?'number':'text'} className="form-input"
      value={String(val||'')} onChange={e => onChange(id, e.target.value)}
      placeholder={placeholder} inputMode={step.type==='number'?'numeric':'text'} autoComplete="off" />
  );
}
