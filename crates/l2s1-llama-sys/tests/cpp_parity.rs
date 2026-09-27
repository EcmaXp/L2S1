//! Opt-in migration comparison against a separately compiled pre-migration bridge.
//! Both implementations must link the same llama.cpp/mtmd build. The reference
//! .so should use -Wl,-Bsymbolic to prevent sd_* symbol interposition.
#![cfg(target_os = "linux")]
use l2s1_llama_sys::*;
use std::ffi::{CStr, CString, c_char, c_void};
struct Library(*mut c_void);
impl Library {
    unsafe fn symbol<T: Copy>(&self, name: &CStr) -> T {
        let p = unsafe { libc::dlsym(self.0, name.as_ptr()) };
        assert!(!p.is_null());
        assert_eq!(size_of::<T>(), size_of::<*mut c_void>());
        unsafe { std::mem::transmute_copy(&p) }
    }
}
impl Drop for Library {
    fn drop(&mut self) {
        unsafe {
            libc::dlclose(self.0);
        }
    }
}
#[test]
#[ignore = "requires L2S1_CPP_REFERENCE shared library and SKID_MODEL; optionally SKID_VISION_MMPROJ"]
fn rust_matches_previous_cpp_bridge() {
    unsafe {
        let path = CString::new(std::env::var("L2S1_CPP_REFERENCE").unwrap()).unwrap();
        let library = Library(libc::dlopen(
            path.as_ptr(),
            libc::RTLD_NOW | libc::RTLD_LOCAL,
        ));
        assert!(
            !library.0.is_null(),
            "{}",
            CStr::from_ptr(libc::dlerror()).to_string_lossy()
        );
        type Open = unsafe extern "C" fn(
            *const c_char,
            u32,
            u32,
            u32,
            i32,
            i32,
            bool,
            *mut c_char,
            usize,
        ) -> *mut c_void;
        type Close = unsafe extern "C" fn(*mut c_void);
        type Forward = unsafe extern "C" fn(
            *mut c_void,
            *const i32,
            i32,
            bool,
            *mut i32,
            *mut f32,
            usize,
            *mut c_char,
            usize,
        ) -> bool;
        type Parallel = unsafe extern "C" fn(
            *mut c_void,
            *const *const i32,
            *const i32,
            i32,
            u32,
            bool,
            *mut i32,
            *mut f32,
            usize,
            *mut c_char,
            usize,
        ) -> bool;
        type Restore = unsafe extern "C" fn(
            *mut c_void,
            *const *const i32,
            *const i32,
            i32,
            usize,
            *mut i32,
            *mut f32,
            usize,
            *mut NativeRestoreMetrics,
            *mut c_char,
            usize,
        ) -> bool;
        let open: Open = library.symbol(c"sd_open");
        let close: Close = library.symbol(c"sd_close");
        let forward: Forward = library.symbol(c"sd_forward");
        let parallel: Parallel = library.symbol(c"sd_forward_parallel");
        let restore: Restore = library.symbol(c"sd_forward_restore");
        let model = CString::new(std::env::var("SKID_MODEL").unwrap()).unwrap();
        let mut error = [0; 1024];
        let rust = sd_open(
            model.as_ptr(),
            2048,
            64,
            64,
            -1,
            4,
            false,
            error.as_mut_ptr(),
            error.len(),
        );
        assert!(
            !rust.is_null(),
            "{}",
            CStr::from_ptr(error.as_ptr()).to_string_lossy()
        );
        let cpp = open(
            model.as_ptr(),
            2048,
            64,
            64,
            -1,
            4,
            false,
            error.as_mut_ptr(),
            error.len(),
        );
        assert!(!cpp.is_null());
        let vocab = sd_vocab_size(rust) as usize;
        let prompt =
            CString::new(format!("{} Answer: A", "A red box is cold. ".repeat(35))).unwrap();
        let mut tokens = vec![0; 2048];
        let count = sd_tokenize(
            rust,
            prompt.as_ptr(),
            prompt.as_bytes().len() as i32,
            true,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
        );
        assert!(count > 64);
        tokens.truncate(count as usize);
        let mut changed = tokens.clone();
        *changed.last_mut().unwrap() = tokens[3];
        for input in [&tokens, &changed, &changed] {
            let mut a = vec![0.0; vocab];
            let mut b = a.clone();
            let (mut ar, mut br) = (0, 0);
            assert!(sd_forward(
                rust,
                input.as_ptr(),
                input.len() as i32,
                true,
                &mut ar,
                a.as_mut_ptr(),
                vocab,
                error.as_mut_ptr(),
                error.len()
            ));
            assert!(forward(
                cpp,
                input.as_ptr(),
                input.len() as i32,
                true,
                &mut br,
                b.as_mut_ptr(),
                vocab,
                error.as_mut_ptr(),
                error.len()
            ));
            assert_eq!(ar, br);
            assert_eq!(a, b, "full vocabulary differs");
        }
        let inputs = [tokens.as_ptr(), changed.as_ptr()];
        let counts = [tokens.len() as i32, changed.len() as i32];
        for dynamic in [false, true] {
            let mut a = vec![0.0; vocab * 2];
            let mut b = a.clone();
            let (mut ar, mut br) = ([0; 2], [0; 2]);
            assert!(
                sd_forward_parallel(
                    rust,
                    inputs.as_ptr(),
                    counts.as_ptr(),
                    2,
                    2,
                    dynamic,
                    ar.as_mut_ptr(),
                    a.as_mut_ptr(),
                    a.len(),
                    error.as_mut_ptr(),
                    error.len()
                ),
                "{}",
                CStr::from_ptr(error.as_ptr()).to_string_lossy()
            );
            assert!(parallel(
                cpp,
                inputs.as_ptr(),
                counts.as_ptr(),
                2,
                2,
                dynamic,
                br.as_mut_ptr(),
                b.as_mut_ptr(),
                b.len(),
                error.as_mut_ptr(),
                error.len()
            ));
            assert_eq!(ar, br);
            assert_eq!(a, b, "parallel vocabulary differs");
        }
        for limit in [0, 64 * 1024 * 1024] {
            let mut a = vec![0.0; vocab * 2];
            let mut b = a.clone();
            let (mut ar, mut br) = ([0; 2], [0; 2]);
            let (mut am, mut bm) = (
                NativeRestoreMetrics::default(),
                NativeRestoreMetrics::default(),
            );
            assert!(sd_forward_restore(
                rust,
                inputs.as_ptr(),
                counts.as_ptr(),
                2,
                limit,
                ar.as_mut_ptr(),
                a.as_mut_ptr(),
                a.len(),
                &mut am,
                error.as_mut_ptr(),
                error.len()
            ));
            assert!(restore(
                cpp,
                inputs.as_ptr(),
                counts.as_ptr(),
                2,
                limit,
                br.as_mut_ptr(),
                b.as_mut_ptr(),
                b.len(),
                &mut bm,
                error.as_mut_ptr(),
                error.len()
            ));
            assert_eq!(
                (am.fallback, am.snapshot_bytes, am.restores),
                (bm.fallback, bm.snapshot_bytes, bm.restores)
            );
            assert_eq!(ar, br);
            assert_eq!(a, b, "snapshot vocabulary differs");
        }
        sd_close(rust);
        close(cpp);
    }
}
