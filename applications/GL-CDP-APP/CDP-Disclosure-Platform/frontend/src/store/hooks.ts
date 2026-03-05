/**
 * GL-CDP-APP v1.0 - Typed Redux Hooks
 *
 * Pre-typed versions of useDispatch and useSelector bound to this
 * application's store shape. Import these throughout the component
 * tree instead of the generic react-redux hooks.
 */

import { useDispatch, useSelector } from 'react-redux';
import type { TypedUseSelectorHook } from 'react-redux';
import type { AppDispatch, AppRootState } from './index';

/** Typed dispatch hook bound to AppDispatch (supports thunks). */
export const useAppDispatch: () => AppDispatch = useDispatch;

/** Typed selector hook bound to AppRootState for full slice inference. */
export const useAppSelector: TypedUseSelectorHook<AppRootState> = useSelector;
