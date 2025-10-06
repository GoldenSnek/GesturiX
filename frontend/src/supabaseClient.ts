import { createClient } from '@supabase/supabase-js';

// Your Supabase project URL and anon key
const SUPABASE_URL = 'https://mqyfdubodreetkwjewch.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1xeWZkdWJvZHJlZXRrd2pld2NoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg1ODc5NjcsImV4cCI6MjA3NDE2Mzk2N30.gJ-dbjHeGGkmXvskastCcVjetZVqnLS8OPqZaOZ4fLI';

// Create a single supabase client for the whole app
export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
